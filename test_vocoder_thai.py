#!/usr/bin/env python3
"""
测试 BigVGAN vocoder 对泰语的支持
用法: python test_vocoder_thai.py <wav_file_path> [--output-dir <dir>]
"""

import argparse
import torch
import torchaudio
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.audio import mel_spectrogram


def load_audio(wav_path: Path, target_sr: int = 22050):
    """加载音频并重采样到目标采样率"""
    waveform, sr = torchaudio.load(str(wav_path))
    
    # 如果是立体声，转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样到目标采样率
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr


def extract_mel_spectrogram(waveform, config):
    """提取梅尔频谱图（使用与训练相同的参数）"""
    mel_fn_args = {
        "n_fft": config.s2mel['preprocess_params']['spect_params']['n_fft'],
        "win_size": config.s2mel['preprocess_params']['spect_params']['win_length'],
        "hop_size": config.s2mel['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config.s2mel['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": config.s2mel["preprocess_params"]["sr"],
        "fmin": config.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    
    mel = mel_spectrogram(waveform, **mel_fn_args)
    return mel


def test_vocoder(wav_path: Path, config_path: Path, output_dir: Path = None, device: str = "cuda"):
    """测试 vocoder 重建音频的质量"""
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    # 加载配置
    config = OmegaConf.load(config_path)
    print(f"[Info] Loaded config from: {config_path}")
    
    # 加载 BigVGAN
    bigvgan_name = config.vocoder.name
    print(f"[Info] Loading BigVGAN: {bigvgan_name}")
    vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    vocoder = vocoder.to(device)
    vocoder.remove_weight_norm()
    vocoder.eval()
    print(f"[Info] BigVGAN loaded successfully")
    
    # 加载原始音频
    print(f"[Info] Loading audio: {wav_path}")
    waveform, sr = load_audio(wav_path, target_sr=config.s2mel["preprocess_params"]["sr"])
    print(f"[Info] Audio loaded: shape={waveform.shape}, sr={sr}")
    
    # 提取梅尔频谱图
    print(f"[Info] Extracting mel spectrogram...")
    mel = extract_mel_spectrogram(waveform, config)
    print(f"[Info] Mel spectrogram shape: {mel.shape}")
    
    # 使用 vocoder 重建音频
    print(f"[Info] Reconstructing audio with BigVGAN...")
    with torch.no_grad():
        mel = mel.to(device)
        # BigVGAN 期望输入格式: (batch, n_mels, time)
        reconstructed = vocoder(mel.float())
        reconstructed = reconstructed.cpu()
    
    print(f"[Info] Reconstructed audio shape: {reconstructed.shape}")
    
    # 归一化音频
    reconstructed = torch.clamp(reconstructed, -1.0, 1.0)
    original = torch.clamp(waveform, -1.0, 1.0)
    
    # 确保音频格式正确：torchaudio.save 期望 (channels, samples) 格式
    # 如果是 3D tensor (batch, channels, samples)，去掉 batch 维度
    if reconstructed.ndim == 3:
        reconstructed = reconstructed.squeeze(0)  # (batch, channels, samples) -> (channels, samples)
    if original.ndim == 3:
        original = original.squeeze(0)
    
    # 如果是单声道且是 1D，转换为 2D (1, samples)
    if reconstructed.ndim == 1:
        reconstructed = reconstructed.unsqueeze(0)
    if original.ndim == 1:
        original = original.unsqueeze(0)
    
    # 保存结果
    if output_dir is None:
        output_dir = wav_path.parent / "vocoder_test_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_path = output_dir / f"{wav_path.stem}_original.wav"
    reconstructed_path = output_dir / f"{wav_path.stem}_reconstructed.wav"
    
    torchaudio.save(str(original_path), original, sr)
    torchaudio.save(str(reconstructed_path), reconstructed, sr)
    
    print(f"[Info] Saved original audio: {original_path}")
    print(f"[Info] Saved reconstructed audio: {reconstructed_path}")
    
    # 计算一些基本统计信息
    print("\n[Info] Audio Statistics:")
    print(f"  Original:")
    print(f"    Duration: {original.shape[-1] / sr:.2f} seconds")
    print(f"    Min: {original.min():.4f}, Max: {original.max():.4f}")
    print(f"    Mean: {original.mean():.4f}, Std: {original.std():.4f}")
    print(f"  Reconstructed:")
    print(f"    Duration: {reconstructed.shape[-1] / sr:.2f} seconds")
    print(f"    Min: {reconstructed.min():.4f}, Max: {reconstructed.max():.4f}")
    print(f"    Mean: {reconstructed.mean():.4f}, Std: {reconstructed.std():.4f}")
    
    # 计算 MSE
    if original.shape == reconstructed.shape:
        mse = torch.mean((original - reconstructed) ** 2).item()
        print(f"\n[Info] Mean Squared Error (MSE): {mse:.6f}")
    else:
        print(f"\n[Warn] Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
        print("       Cannot compute MSE directly")
    
    print(f"\n[Info] Test completed! Please listen to the audio files to assess quality.")
    print(f"       Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test BigVGAN vocoder on Thai audio")
    parser.add_argument("wav_file", type=Path, help="Path to Thai audio WAV file")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="Path to config file (default: checkpoints/config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for test results (default: <wav_file_dir>/vocoder_test_output)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    if not args.wav_file.exists():
        print(f"[Error] WAV file not found: {args.wav_file}")
        return
    
    if not args.config.exists():
        print(f"[Error] Config file not found: {args.config}")
        return
    
    test_vocoder(args.wav_file, args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()

