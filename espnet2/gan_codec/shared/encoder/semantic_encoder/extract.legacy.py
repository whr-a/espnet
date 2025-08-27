import torch
import torchaudio
import torchaudio.transforms as T

# -------------------- Main Function (Corrected Version) --------------------

def get_canary_flash_preprocessor_output(
    input_signal_24k: torch.Tensor, length_24k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    一个独立的函数，用于将 24kHz 的原始音频波形处理成 nvidia/canary-1b-flash 模型所需的输入特征。

    该函数严格遵循 Hugging Face 上的 `preprocessor_config.json` 文件配置。
    完整的预处理流程包括：
    1. 从 24kHz 到 16kHz 的重采样。
    2. 预加重滤波 (Pre-emphasis)。
    3. 计算对数梅尔频谱图 (n_mels=128, norm='slaney')。
    4. 特征归一化 (per_feature)。

    Args:
        input_signal_24k (torch.Tensor): 形状为 (B, T) 的 24kHz 原始音频波形张量。
        length_24k (torch.Tensor): 形状为 (B,) 的张量，包含每个 24kHz 音频的实际长度。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - processed_signal (torch.Tensor): 处理后的对数梅尔频谱图，形状为 (B, 128, T')。
        - processed_signal_length (torch.Tensor): 处理后频谱图的实际长度，形状为 (B,)。
    """
    # ------------------- Hard-coded Parameters from preprocessor_config.json -------------------
    orig_sample_rate = 24000
    target_sample_rate = 16000
    n_fft = 500
    win_length_samples = 400
    hop_length_samples = 160
    n_mels = 128
    preemph_coeff = 0.97
    mel_norm_type = "slaney"
    mag_power = 2.0
    log_guard_val = 2**-24
    # -----------------------------------------------------------------------------------------

    device = input_signal_24k.device

    # 1. 重采样 (Resampling)
    resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate).to(device)
    resampled_signal_16k = resampler(input_signal_24k)
    
    resample_ratio = target_sample_rate / orig_sample_rate
    length_16k = torch.round(length_24k.float() * resample_ratio).long()

    # 2. 预加重 (Pre-emphasis)
    preemphasized_signal = torch.cat(
        (resampled_signal_16k[:, :1], resampled_signal_16k[:, 1:] - preemph_coeff * resampled_signal_16k[:, :-1]),
        dim=1,
    )

    # 3. 计算梅尔频谱图
    featurizer = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        win_length=win_length_samples,
        hop_length=hop_length_samples,
        n_fft=n_fft,
        n_mels=n_mels,
        norm=mel_norm_type,
        window_fn=torch.hann_window,
        power=mag_power,
        center=True,
    ).to(device)
    
    mel_spec = featurizer(preemphasized_signal)

    # 4. 取对数
    log_mel_spec = torch.log(mel_spec + log_guard_val)

    # 5. 特征归一化 (per_feature)
    processed_signal = torch.zeros_like(log_mel_spec)
    
    # ============================ THIS LINE IS CORRECTED ============================
    # 正确的公式，当 center=True 时
    processed_signal_length = torch.div(length_16k, hop_length_samples, rounding_mode='floor') + 1
    # ===============================================================================
    
    for i in range(log_mel_spec.shape[0]):
        current_len = processed_signal_length[i].item()
        if current_len <= 0:
            continue
        
        valid_spec = log_mel_spec[i, :, :current_len]
        mean = torch.mean(valid_spec, dim=1, keepdim=True)
        std = torch.std(valid_spec, dim=1, keepdim=True)
        
        normalized_spec = (valid_spec - mean) / (std + 1e-5)
        processed_signal[i, :, :current_len] = normalized_spec

    return processed_signal, processed_signal_length

# -------------------- Example Usage (now with correct output) --------------------
if __name__ == '__main__':
    batch_size = 4
    sample_rate_24k = 24000
    
    max_len = 3523453
    audio_lengths_24k = torch.tensor([3523453, 23414, 55242, 99999]) # 使用您提供的数据
    
    signals_24k = torch.randn(batch_size, max_len)
    for i in range(batch_size):
        signals_24k[i, audio_lengths_24k[i]:] = 0.0
        
    print(f"传入信号形状 (24kHz): {signals_24k.shape}")
    print(f"传入信号长度 (24kHz): {audio_lengths_24k}")
    print("-" * 30)
    
    processed_features, processed_lengths = get_canary_flash_preprocessor_output(signals_24k, audio_lengths_24k)
    
    print("修正后的结果:")
    print(f"最终特征形状: {processed_features.shape}")
    print(f"最终特征长度: {processed_lengths}")

    # 验证对齐
    assert processed_features.shape[-1] == torch.max(processed_lengths).item()
    print("\n验证通过：特征形状的最大长度与计算出的最大长度一致！")