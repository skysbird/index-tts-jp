#!/usr/bin/env python3
"""
增量提取语义特征 (feat) 并更新 manifest。

这个脚本：
1. 读取已有的 manifest（包含 codes, condition, emo_vec 等）
2. 从 audio_path 提取 feat
3. 保存 feat 到文件
4. 更新 manifest 添加 feat_path 字段

用法：
    python tools/extract_feat_incremental.py \
        --manifest th_processed_data/gpt_pairs_train.jsonl \
        --output-dir th_processed_data \
        --config checkpoints/config_finetune.yaml \
        --audio-root /data/sky/dataset-maker/datasets_folder/th/
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from indextts.utils.maskgct_utils import build_semantic_model
from transformers import SeamlessM4TFeatureExtractor


class SemanticExtractor:
    def __init__(self, stats_path: Path, device: torch.device):
        self.device = device
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=stats_path
        )
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.semantic_model.eval()

    @torch.inference_mode()
    def extract(
        self,
        waveforms: List[torch.Tensor],
        sample_rates: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arrays: List[np.ndarray] = []
        for wav, sr in zip(waveforms, sample_rates):
            current = wav
            if sr != 16000:
                current = torchaudio.functional.resample(current, sr, 16000)
            arrays.append(current.squeeze(0).cpu().numpy())

        inputs = self.feature_extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 使用混合精度加速（如果 GPU 可用）
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.semantic_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                feat = outputs.hidden_states[17]
                feat = (feat - self.semantic_mean) / self.semantic_std
        else:
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = outputs.hidden_states[17]
            feat = (feat - self.semantic_mean) / self.semantic_std
        
        return feat, attention_mask


def load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def resolve_audio_path(audio_value: str, audio_roots: List[Path]) -> Optional[Path]:
    path = Path(audio_value).expanduser()
    if path.is_file():
        return path

    audio_rel = Path(audio_value)
    for root in audio_roots:
        candidate = (root / audio_rel).expanduser()
        if candidate.is_file():
            return candidate
    return None


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def process_batch(
    records: List[Dict],
    semantic_extractor: SemanticExtractor,
    feat_dir: Path,
    output_root: Path,
    audio_roots: List[Path],
    executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[List[Dict], int]:
    """处理一批记录，提取 feat 并更新 manifest entry"""
    
    prepared = []
    skipped = 0

    # 判断是否是 paired manifest
    is_paired = any("prompt_audio_path" in r and "target_audio_path" in r for r in records)

    # 准备音频路径
    for record in records:
        # 判断是 paired 还是 single manifest
        if is_paired:
            # Paired manifest: 需要提取 prompt_feat 和 target feat
            prompt_audio_path_str = record.get("prompt_audio_path", "")
            target_audio_path_str = record.get("target_audio_path", "")
            
            if not prompt_audio_path_str or not target_audio_path_str:
                print(f"[Warn] Missing audio_path for paired record {record.get('id', 'unknown')}")
                skipped += 1
                continue
            
            prompt_resolved = resolve_audio_path(prompt_audio_path_str, audio_roots)
            target_resolved = resolve_audio_path(target_audio_path_str, audio_roots)
            
            if prompt_resolved is None:
                print(f"[Warn] Prompt audio file not found: {prompt_audio_path_str}")
                skipped += 1
                continue
            if target_resolved is None:
                print(f"[Warn] Target audio file not found: {target_audio_path_str}")
                skipped += 1
                continue
            
            prepared.append({
                "record": record,
                "prompt_audio_path": prompt_resolved,
                "target_audio_path": target_resolved,
                "is_paired": True,
            })
        else:
            # Single manifest: 只需要一个 audio_path
            audio_path_str = record.get("audio_path", "")
            
            if not audio_path_str:
                print(f"[Warn] No audio_path found for record {record.get('id', 'unknown')}")
                skipped += 1
                continue

            resolved = resolve_audio_path(audio_path_str, audio_roots)
            if resolved is None:
                print(f"[Warn] Audio file not found: {audio_path_str}")
                skipped += 1
                continue

            prepared.append({
                "record": record,
                "audio_path": resolved,
                "is_paired": False,
            })

    if not prepared:
        return [], skipped

    # 加载音频
    if executor is not None:
        futures: Dict[Future, Dict] = {}
        for item in prepared:
            if item["is_paired"]:
                # Paired: 加载两个音频
                future_prompt = executor.submit(load_audio, item["prompt_audio_path"], 24000)
                future_target = executor.submit(load_audio, item["target_audio_path"], 24000)
                futures[future_prompt] = (item, "prompt")
                futures[future_target] = (item, "target")
            else:
                # Single: 加载一个音频
                future = executor.submit(load_audio, item["audio_path"], 24000)
                futures[future] = (item, "single")
        
        for future, (item, audio_type) in futures.items():
            try:
                waveform, sr = future.result()
            except Exception:
                traceback.print_exc()
                if audio_type == "single":
                    skipped += 1
                continue
            
            if audio_type == "prompt":
                item["prompt_waveform"] = waveform
                item["prompt_sr"] = sr
            elif audio_type == "target":
                item["target_waveform"] = waveform
                item["target_sr"] = sr
            else:  # single
                item["waveform"] = waveform
                item["sr"] = sr
    else:
        for item in prepared:
            try:
                if item["is_paired"]:
                    prompt_waveform, prompt_sr = load_audio(item["prompt_audio_path"], target_sr=24000)
                    target_waveform, target_sr = load_audio(item["target_audio_path"], target_sr=24000)
                    item["prompt_waveform"] = prompt_waveform
                    item["prompt_sr"] = prompt_sr
                    item["target_waveform"] = target_waveform
                    item["target_sr"] = target_sr
                else:
                    waveform, sr = load_audio(item["audio_path"], target_sr=24000)
                    item["waveform"] = waveform
                    item["sr"] = sr
            except Exception:
                traceback.print_exc()
                skipped += 1
                continue

    # 过滤掉加载失败的
    if is_paired:
        prepared = [item for item in prepared if "prompt_waveform" in item and "target_waveform" in item]
    else:
        prepared = [item for item in prepared if "waveform" in item]
    
    if not prepared:
        return [], skipped

    # 批量提取特征（优化：一次性处理整个 batch）
    updated_records = []
    
    if is_paired:
        # Paired manifest: 批量提取 prompt_feat 和 target feat
        prompt_waveforms = [item["prompt_waveform"] for item in prepared]
        prompt_sample_rates = [item["prompt_sr"] for item in prepared]
        target_waveforms = [item["target_waveform"] for item in prepared]
        target_sample_rates = [item["target_sr"] for item in prepared]
        
        # 批量提取 prompt_feat
        prompt_feat, _ = semantic_extractor.extract(prompt_waveforms, prompt_sample_rates)
        prompt_feat_np = prompt_feat.detach().cpu().numpy().astype(np.float32)
        
        # 批量提取 target feat
        target_feat, _ = semantic_extractor.extract(target_waveforms, target_sample_rates)
        target_feat_np = target_feat.detach().cpu().numpy().astype(np.float32)
        
        # 保存并更新记录
        for idx, item in enumerate(prepared):
            record = item["record"]
            
            # 保存 prompt_feat（使用 prompt_id 作为文件名）
            prompt_id = record.get("prompt_id", record.get("id", "").split("__")[1] if "__" in record.get("id", "") else "")
            if prompt_id:
                prompt_feat_path = feat_dir / f"{prompt_id}.npy"
                save_numpy(prompt_feat_path, prompt_feat_np[idx])
            
            # 保存 target feat（使用 target_id 或 id 作为文件名）
            target_id = record.get("target_id", record.get("id", "").split("__")[0] if "__" in record.get("id", "") else record.get("id", ""))
            target_feat_path = feat_dir / f"{target_id}.npy"
            save_numpy(target_feat_path, target_feat_np[idx])
            
            # 更新 record（保持与 build_gpt_prompt_pairs.py 生成的格式一致）
            updated_record = record.copy()
            if prompt_id:
                # 确保 prompt_feat_path 是绝对路径，然后计算相对路径
                prompt_feat_path_abs = Path(prompt_feat_path).resolve()
                updated_record["prompt_feat_path"] = prompt_feat_path_abs.relative_to(output_root).as_posix()
                updated_record["prompt_feat_len"] = int(prompt_feat_np[idx].shape[0])
            # target 的 feat_path（保持与现有格式一致，使用 feat_path 而不是 target_feat_path）
            target_feat_path_abs = Path(target_feat_path).resolve()
            updated_record["feat_path"] = target_feat_path_abs.relative_to(output_root).as_posix()
            updated_record["feat_len"] = int(target_feat_np[idx].shape[0])
            
            updated_records.append(updated_record)
    else:
        # Single manifest: 批量提取 feat
        waveforms = [item["waveform"] for item in prepared]
        sample_rates = [item["sr"] for item in prepared]
        
        # 批量提取特征
        feat, _ = semantic_extractor.extract(waveforms, sample_rates)
        feat_np = feat.detach().cpu().numpy().astype(np.float32)
        
        # 保存并更新记录
        for idx, item in enumerate(prepared):
            record = item["record"]
            uid = record["id"]
            
            feat_path = feat_dir / f"{uid}.npy"
            save_numpy(feat_path, feat_np[idx])
            
            # 更新 record
            updated_record = record.copy()
            # 确保 feat_path 是绝对路径
            feat_path_abs = Path(feat_path).resolve()
            updated_record["feat_path"] = feat_path_abs.relative_to(output_root).as_posix()
            updated_record["feat_len"] = int(feat_np[idx].shape[0])
            
            updated_records.append(updated_record)

    return updated_records, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="增量提取语义特征 (feat) 并更新 manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="输入 manifest 路径（JSONL）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（feat 文件将保存在 output_dir/feat/）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="IndexTTS config YAML（用于获取 w2v_stat 路径）",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        action="append",
        default=[],
        help="音频文件根目录（可指定多个）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小（建议 32-64，GPU 内存允许的话可以更大）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="并行加载音频的线程数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda/cpu）",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="输出 manifest 路径（默认：覆盖原文件）",
    )

    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    stats_value = OmegaConf.select(cfg, "w2v_stat")
    stats_path = Path(stats_value or "checkpoints/wav2vec2bert_stats.pt")
    if not stats_path.is_absolute():
        stats_path = (args.config.parent / stats_path).resolve()

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # 初始化
    device = torch.device(args.device)
    semantic_extractor = SemanticExtractor(stats_path, device)
    
    # 确保 output_root 和 feat_dir 都是绝对路径
    output_root = args.output_dir.expanduser().resolve()
    feat_dir = (output_root / "feat").resolve()
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    audio_roots = [Path(r).expanduser().resolve() for r in args.audio_root]

    # 读取 manifest
    print(f"[Info] Reading manifest: {args.manifest}")
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # 跳过已经有 feat_path 的记录
            if "feat_path" in record:
                print(f"[Info] Record {record.get('id')} already has feat_path, skipping")
                continue
            records.append(record)

    print(f"[Info] Found {len(records)} records to process")

    # 处理记录
    updated_records = []
    skipped_total = 0
    
    executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    
    try:
        for i in tqdm(range(0, len(records), args.batch_size), desc="Processing batches"):
            batch = records[i:i + args.batch_size]
            updated_batch, skipped = process_batch(
                batch,
                semantic_extractor,
                feat_dir,
                output_root,
                audio_roots,
                executor,
            )
            updated_records.extend(updated_batch)
            skipped_total += skipped
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # 保存更新后的 manifest
    output_manifest = args.output_manifest or args.manifest
    print(f"[Info] Writing updated manifest: {output_manifest}")
    
    # 如果输出文件不同，需要读取原文件并合并
    if output_manifest != args.manifest:
        # 读取原文件中已有 feat_path 的记录
        existing_records = []
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "feat_path" in record:
                    existing_records.append(record)
        
        # 合并所有记录
        all_records = existing_records + updated_records
        # 按 id 排序（如果需要保持顺序）
        all_records.sort(key=lambda x: x.get("id", ""))
    else:
        # 读取原文件，更新记录
        all_records = []
        record_dict = {r["id"]: r for r in updated_records}
        
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # 如果记录已更新，使用更新后的版本
                if record["id"] in record_dict:
                    all_records.append(record_dict[record["id"]])
                else:
                    all_records.append(record)
    
    # 写入更新后的 manifest
    with open(output_manifest, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[Info] Updated {len(updated_records)} records")
    print(f"[Info] Skipped {skipped_total} records")
    print(f"[Info] Total records in manifest: {len(all_records)}")
    print(f"[Info] Done! Updated manifest saved to: {output_manifest}")


if __name__ == "__main__":
    main()

