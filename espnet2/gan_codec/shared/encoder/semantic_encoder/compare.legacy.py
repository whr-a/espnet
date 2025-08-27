import torch
import torchaudio.transforms as T
from nemo.collections.asr.models import EncDecMultiTaskModel
import warnings

# 忽略 NeMo 加载模型时的一些警告信息
warnings.filterwarnings("ignore", category=UserWarning)

# 从我们之前创建的文件中导入我们自己的 SemanticEncoder
from encoder import SemanticEncoder

def run_comparison():
    """
    执行 NeMo 官方模型编码器与我们独立实现的 SemanticEncoder 之间的最终对照实验。
    """
    # --- 0. 环境设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"实验将在设备上运行: {device}")

    # --- 1. 准备一份完全相同的 24kHz 输入数据 ---
    print("\n--- [Step 1] 准备输入数据 ---")
    batch_size = 2
    sample_rate_24k = 24000
    seconds = 4
    max_len = sample_rate_24k * seconds
    
    audio_lengths_24k = torch.tensor([max_len, int(max_len * 0.75)])
    signals_24k = torch.randn(batch_size, max_len)
    
    for i in range(batch_size):
        signals_24k[i, audio_lengths_24k[i]:] = 0.0
        
    print(f"创建了一批 24kHz 音频数据:")
    print(f"  - 形状: {signals_24k.shape}")
    print(f"  - 长度: {audio_lengths_24k.tolist()}")
    
    # 统一将所有输入数据移动到目标设备
    signals_24k_device = signals_24k.to(device)
    audio_lengths_24k_device = audio_lengths_24k.to(device)

    # --- 2. 运行方法一：NeMo 官方模型 (手动分步) ---
    print("\n--- [Step 2] 运行 NeMo 官方模型 ---")
    print("正在加载 nvidia/canary-1b-flash...")
    nemo_canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash').to(device)
    nemo_canary_model.eval()
    print("NeMo 模型加载完毕。")

    print("手动执行 NeMo 的 预处理 -> 编码 流程...")
    with torch.no_grad():
        # NeMo的 preprocessor 模块期望输入已经是 16kHz，所以我们必须在此之前手动重采样。
        # a. 手动重采样
        resampler = T.Resample(orig_freq=24000, new_freq=16000).to(device)
        signals_16k = resampler(signals_24k_device)
        resample_ratio = 16000 / 24000
        audio_lengths_16k = (audio_lengths_24k_device.float() * resample_ratio).long()
        
        # b. 预处理 (输入的是重采样后的 16kHz 音频)
        processed_signal, processed_signal_length = nemo_canary_model.preprocessor(
            input_signal=signals_16k, length=audio_lengths_16k
        )
        # c. 编码
        nemo_encoded, nemo_lengths = nemo_canary_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

    print("NeMo Encoder 输出已生成。")
    print(f"  - 输出形状: {nemo_encoded.shape}, 输出长度: {nemo_lengths.tolist()}")

    # --- 3. 运行方法二：我们自己的独立模型 (端到端) ---
    print("\n--- [Step 3] 运行我们自己的独立模型 ---")
    standalone_encoder = SemanticEncoder().to(device)
    encoder_weights_path = '/u/hwang41/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt'
    standalone_encoder.load_weights(encoder_weights_path, strict=False)
    standalone_encoder.eval()
    
    with torch.no_grad():
        # 我们的 SemanticEncoder 类被设计为端到端模型，
        # 其 .forward() 方法内部已经包含了从 24kHz 开始的所有处理步骤（包括重采样）。
        # 因此，我们直接将 24kHz 的原始音频喂给它。
        standalone_outputs = standalone_encoder(signals_24k_device, audio_lengths_24k_device)
        
        # 为了比较，我们需要将最终 (B, T, D) 的输出转置回 (B, D, T)
        standalone_encoded = standalone_outputs['embeddings']
        standalone_lengths = standalone_outputs['lengths']

    print("独立模型输出已生成。")
    print(f"  - 输出形状: {standalone_encoded.shape}, 输出长度: {standalone_lengths.tolist()}")
    
    # --- 4. 比较两个方法的输出 ---
    print("\n--- [Step 4] 比较两个输出的结果 ---")
    shapes_match = (nemo_encoded.shape == standalone_encoded.shape)
    print(f"1. 输出形状是否一致? \t{'✅ 是' if shapes_match else '❌ 否'}")
    
    lengths_match = torch.equal(nemo_lengths, standalone_lengths)
    print(f"2. 输出长度是否一致? \t{'✅ 是' if lengths_match else '❌ 否'}")

    if shapes_match and lengths_match:
        tolerance = 1e-5
        values_are_close = torch.allclose(nemo_encoded, standalone_encoded, atol=tolerance)
        print(f"3. 数值是否足够接近 (atol={tolerance})? \t{'✅ 是' if values_are_close else '❌ 否'}")
        if not values_are_close:
            abs_diff = torch.abs(nemo_encoded - standalone_encoded)
            max_diff, mean_diff = torch.max(abs_diff).item(), torch.mean(abs_diff).item()
            print(f"   - 最大绝对差值: {max_diff:.8f}")
            print(f"   - 平均绝对差值: {mean_diff:.8f}")
    
    print("\n--- 对照实验完成 ---")

if __name__ == '__main__':
    run_comparison()