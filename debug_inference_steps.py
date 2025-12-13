#!/usr/bin/env python3
"""
分步验证推理流程，定位噪音来源
用法: python debug_inference_steps.py <text> <spk_audio_prompt> [--output-dir <dir>]
"""

import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from indextts.infer_v2_modded import IndexTTS2
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.audio import mel_spectrogram


def save_mel_spectrogram(mel, path, title="Mel Spectrogram"):
    """保存梅尔频谱图为图片"""
    mel_np = mel.squeeze().cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('Time frames')
    plt.ylabel('Mel bins')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved mel spectrogram: {path}")


def mel_to_audio_via_vocoder(mel, vocoder, device_obj, sr=22050):
    """使用 vocoder 将梅尔频谱图转换为音频（用于测试）"""
    with torch.no_grad():
        mel = mel.to(device_obj)
        audio = vocoder(mel.float())
        audio = audio.cpu()
        # 处理维度
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return audio


def debug_inference_steps(
    text: str,
    spk_audio_prompt: str,
    config_path: Path,
    model_dir: Path,
    output_dir: Path,
    emo_audio_prompt: str = None,
    device: str = "cuda"
):
    """分步调试推理流程"""
    
    # 确定设备字符串
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_str = device
    device_obj = torch.device(device_str)
    print(f"[Info] Using device: {device_str}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("[Info] Loading IndexTTS2 model...")
    tts = IndexTTS2(
        cfg_path=str(config_path),
        model_dir=str(model_dir),
        use_fp16=False,
        device=device_str
    )
    print("[Info] Model loaded successfully")
    
    # 加载 vocoder（用于测试中间步骤的梅尔频谱图）
    print("[Info] Loading BigVGAN for intermediate testing...")
    config = OmegaConf.load(config_path)
    vocoder = bigvgan.BigVGAN.from_pretrained(config.vocoder.name, use_cuda_kernel=False)
    vocoder = vocoder.to(device_obj)
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("[Info] BigVGAN loaded")
    
    # ========== Step 1: 文本处理 ==========
    print("\n" + "="*80)
    print("Step 1: Text Processing")
    print("="*80)
    text_tokens = tts.tokenizer.tokenize(text)
    print(f"  Text: {text}")
    print(f"  Text tokens: {text_tokens}")
    print(f"  Token count: {len(text_tokens)}")
    
    # 保存文本信息
    with open(output_dir / "step1_text_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Text: {text}\n")
        f.write(f"Tokens: {text_tokens}\n")
        f.write(f"Token count: {len(text_tokens)}\n")
    
    # ========== Step 2: 提取 conditioning ==========
    print("\n" + "="*80)
    print("Step 2: Extract Conditioning from Audio Prompt")
    print("="*80)
    
    # 加载参考音频
    audio, sr = torchaudio.load(spk_audio_prompt)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # 重采样（保持在 CPU 上，因为 extract_features 需要 CPU）
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
    
    # 提取 conditioning
    # 使用正确的方法提取语义特征（extract_features 需要 CPU 上的 numpy 数组）
    inputs = tts.extract_features(audio_16k.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device_obj)
    attention_mask = inputs["attention_mask"].to(device_obj)
    spk_cond_emb = tts.get_emb(input_features, attention_mask)
    
    # 量化得到 S_ref
    _, S_ref = tts.semantic_codec.quantize(spk_cond_emb)
    
    # 将音频移到 GPU 用于后续处理
    audio_22k = audio_22k.to(device_obj)
    audio_16k_gpu = audio_16k.to(device_obj)
    
    ref_mel = tts.mel_fn(audio_22k.float())
    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device_obj)
    
    # 提取 fbank 特征用于 style encoder
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k_gpu.float(),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0).to(device_obj)
    
    prompt_condition = tts.s2mel.models['length_regulator'](
        S_ref,
        ylens=ref_target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]
    
    style = tts.campplus_model(feat)  # 使用 campplus_model 提取 style
    
    # 提取 speaker conditioning
    # spk_cond_emb 是 (batch, time, dim) 格式，这是语义特征 embedding
    # 不需要单独调用 get_conditioning，因为 merge_emovec 和 inference_speech 会内部处理
    
    print(f"  ref_mel shape: {ref_mel.shape}")
    print(f"  spk_cond_emb (semantic) shape: {spk_cond_emb.shape}")
    print(f"  style shape: {style.shape}")
    print(f"  prompt_condition shape: {prompt_condition.shape}")
    
    # 保存参考音频的梅尔频谱图
    save_mel_spectrogram(ref_mel, output_dir / "step2_ref_mel.png", "Reference Audio Mel Spectrogram")
    
    # 测试：用 vocoder 重建参考音频（验证 vocoder 质量）
    ref_audio_reconstructed = mel_to_audio_via_vocoder(ref_mel, vocoder, device_obj)
    torchaudio.save(str(output_dir / "step2_ref_audio_reconstructed.wav"), ref_audio_reconstructed, 22050)
    print(f"  Saved reconstructed reference audio: {output_dir / 'step2_ref_audio_reconstructed.wav'}")
    
    # ========== Step 3: GPT 生成语义 codes ==========
    print("\n" + "="*80)
    print("Step 3: GPT Generate Semantic Codes")
    print("="*80)
    
    text_tokens_tensor = torch.tensor(
        tts.tokenizer.convert_tokens_to_ids(text_tokens),
        dtype=torch.int32,
        device=device_obj
    ).unsqueeze(0)
    
    # 参考 infer_v2_modded.py 第577-583行和592-597行
    # merge_emovec 直接传入 (batch, time, dim) 格式，内部会处理转置
    # 注意：cond_lengths 在 infer_v2_modded.py 中使用的是 shape[-1]（即 dim），这看起来不对
    # 但为了保持一致，我们也使用 shape[-1]
    emovec = tts.gpt.merge_emovec(
        spk_cond_emb,  # (batch, time, dim) - 直接传入，内部会转置
        spk_cond_emb,  # 使用相同的作为 emo_speech_conditioning_latent
        torch.tensor([spk_cond_emb.shape[-1]], device=device_obj),  # 使用 shape[-1] 保持与 infer_v2_modded.py 一致
        torch.tensor([spk_cond_emb.shape[-1]], device=device_obj),
        alpha=1.0
    )
    
    with torch.no_grad():
        # inference_speech 直接传入 (batch, time, dim) 格式（参考 infer_v2_modded.py 第592-597行）
        # 内部会处理转置，cond_lengths 使用 shape[-1]（虽然看起来不对，但为了保持一致）
        codes, speech_conditioning_latent = tts.gpt.inference_speech(
            spk_cond_emb,  # (batch, time, dim) - 直接传入，内部会转置
            text_tokens_tensor,
            spk_cond_emb,  # 使用相同的作为 emo_speech_condition
            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=device_obj),  # 使用 shape[-1] 保持与 infer_v2_modded.py 一致
            emo_cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=device_obj),
            emo_vec=emovec,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            num_return_sequences=1,
            max_generate_length=1500,
        )
    
    # 处理 codes
    code_len = codes.shape[-1]
    if tts.stop_mel_token in codes[0]:
        stop_idx = (codes[0] == tts.stop_mel_token).nonzero(as_tuple=False)
        if len(stop_idx) > 0:
            code_len = stop_idx[0].item()
    codes = codes[:, :code_len]
    
    print(f"  Generated codes shape: {codes.shape}")
    print(f"  Codes length: {code_len}")
    print(f"  Codes min: {codes.min()}, max: {codes.max()}")
    print(f"  Codes unique values: {len(torch.unique(codes))}")
    
    # 保存 codes 统计信息
    codes_np = codes.cpu().numpy()
    np.save(output_dir / "step3_codes.npy", codes_np)
    with open(output_dir / "step3_codes_info.txt", "w") as f:
        f.write(f"Codes shape: {codes.shape}\n")
        f.write(f"Codes length: {code_len}\n")
        f.write(f"Codes min: {codes.min().item()}, max: {codes.max().item()}\n")
        f.write(f"Unique codes: {len(torch.unique(codes))}\n")
        f.write(f"Codes distribution:\n")
        unique, counts = torch.unique(codes, return_counts=True)
        for u, c in zip(unique[:20], counts[:20]):  # 只显示前20个
            f.write(f"  Code {u.item()}: {c.item()} times\n")
    
    # ========== Step 4: Codes -> 语义特征 ==========
    print("\n" + "="*80)
    print("Step 4: Codes to Semantic Features")
    print("="*80)
    
    with torch.no_grad():
        S_infer = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
    
    print(f"  S_infer shape: {S_infer.shape}")
    print(f"  S_infer min: {S_infer.min():.4f}, max: {S_infer.max():.4f}")
    print(f"  S_infer mean: {S_infer.mean():.4f}, std: {S_infer.std():.4f}")
    
    # 保存语义特征
    np.save(output_dir / "step4_semantic_features.npy", S_infer.cpu().numpy())
    
    # ========== Step 5: GPT Forward ==========
    print("\n" + "="*80)
    print("Step 5: GPT Forward Pass")
    print("="*80)
    
    code_lens = torch.LongTensor([codes.shape[-1]]).to(device_obj)
    use_speed = torch.zeros(spk_cond_emb.size(0)).to(device_obj).long()
    
    with torch.no_grad():
        latent = tts.gpt(
            speech_conditioning_latent,
            text_tokens_tensor,
            torch.tensor([text_tokens_tensor.shape[-1]], device=device_obj),
            codes,
            code_lens,
            spk_cond_emb.transpose(1, 2),  # emo_speech_condition
            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device_obj),
            emo_cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device_obj),
            emo_vec=emovec,
            use_speed=use_speed,
        )
        latent = tts.s2mel.models['gpt_layer'](latent)
        S_infer = S_infer + latent
    
    print(f"  Latent shape: {latent.shape}")
    print(f"  S_infer (after adding latent) shape: {S_infer.shape}")
    
    # ========== Step 6: Length Regulator ==========
    print("\n" + "="*80)
    print("Step 6: Length Regulator")
    print("="*80)
    
    target_lengths = (code_lens * 1.72).long()
    
    with torch.no_grad():
        cond = tts.s2mel.models['length_regulator'](
            S_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]
    
    print(f"  cond shape: {cond.shape}")
    
    # ========== Step 7: S2Mel 生成梅尔频谱图 ==========
    print("\n" + "="*80)
    print("Step 7: S2Mel Generate Mel Spectrogram")
    print("="*80)
    
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    if style.dim() == 1:
        style = style.unsqueeze(0)
    
    mel_lengths = torch.full(
        (cat_condition.size(0),),
        cat_condition.size(1),
        dtype=torch.long,
        device=cond.device,
    )
    
    with torch.no_grad():
        # cfm.inference 的参数顺序：mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5
        vc_target = tts.s2mel.models['cfm'].inference(
            cat_condition,  # mu
            mel_lengths,  # x_lens
            ref_mel,  # prompt
            style,  # style
            None,  # f0
            25,  # n_timesteps (diffusion_steps)
            temperature=1.0,  # temperature
            inference_cfg_rate=0.7,  # inference_cfg_rate
        )
        vc_target = vc_target[:, :, ref_mel.size(-1):]
    
    print(f"  Generated mel spectrogram shape: {vc_target.shape}")
    print(f"  Mel min: {vc_target.min():.4f}, max: {vc_target.max():.4f}")
    print(f"  Mel mean: {vc_target.mean():.4f}, std: {vc_target.std():.4f}")
    
    # 保存生成的梅尔频谱图
    save_mel_spectrogram(vc_target, output_dir / "step7_generated_mel.png", "Generated Mel Spectrogram")
    
    # 测试：用 vocoder 转换生成的梅尔频谱图（这是关键测试！）
    print("\n  Testing generated mel with vocoder...")
    generated_audio = mel_to_audio_via_vocoder(vc_target, vocoder, device_obj)
    print(f"  Generated audio shape: {generated_audio.shape}")
    print(f"  Generated audio range: min={generated_audio.min():.4f}, max={generated_audio.max():.4f}, abs_max={generated_audio.abs().max():.4f}")
    torchaudio.save(str(output_dir / "step7_generated_audio_from_mel.wav"), generated_audio, 22050)
    print(f"  Saved audio from generated mel: {output_dir / 'step7_generated_audio_from_mel.wav'}")
    print(f"  ⚠️  This is the key test! If this audio has noise, the problem is in S2Mel or earlier steps.")
    
    # ========== Step 8: 完整推理（对比） ==========
    print("\n" + "="*80)
    print("Step 8: Full Inference (for comparison)")
    print("="*80)
    
    full_output_path = output_dir / "step8_full_inference.wav"
    tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=str(full_output_path),
        emo_audio_prompt=emo_audio_prompt,
        verbose=False
    )
    print(f"  Saved full inference result: {full_output_path}")
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"All intermediate results saved to: {output_dir}")
    print("\nKey files to check:")
    print("  1. step2_ref_audio_reconstructed.wav - Vocoder quality test (should be clean)")
    print("  2. step7_generated_audio_from_mel.wav - Generated mel quality test (KEY TEST!)")
    print("  3. step8_full_inference.wav - Full pipeline result")
    print("\nIf step7 has noise but step2 is clean:")
    print("  → Problem is in GPT codes generation or S2Mel")
    print("\nIf step7 is clean but step8 has noise:")
    print("  → Problem is in vocoder or final processing")
    print("\nCompare the audio files to identify where noise is introduced!")


def main():
    parser = argparse.ArgumentParser(description="Debug inference steps to locate noise source")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("spk_audio_prompt", type=Path, help="Speaker audio prompt")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="Config file path"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_inference_output"),
        help="Output directory for debug results"
    )
    parser.add_argument(
        "--emo-audio-prompt",
        type=Path,
        default=None,
        help="Emotion audio prompt (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    if not args.spk_audio_prompt.exists():
        print(f"[Error] Speaker audio prompt not found: {args.spk_audio_prompt}")
        return
    
    if not args.config.exists():
        print(f"[Error] Config file not found: {args.config}")
        return
    
    debug_inference_steps(
        text=args.text,
        spk_audio_prompt=str(args.spk_audio_prompt),
        config_path=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        emo_audio_prompt=str(args.emo_audio_prompt) if args.emo_audio_prompt else None,
        device=args.device
    )


if __name__ == "__main__":
    main()

