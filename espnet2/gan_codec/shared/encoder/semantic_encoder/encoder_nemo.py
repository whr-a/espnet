import torch
from omegaconf import OmegaConf, DictConfig, open_dict
from nemo.core.classes import NeuralModule
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, AcousticEncodedRepresentation

class SemanticEncoder(NeuralModule):
    def __init__(self, cfg_path: str):
        super().__init__()
        cfg: Union[DictConfig, Any] = OmegaConf.load(cfg_path)
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self._cfg = cfg

        self.preprocessor = AudioToMelSpectrogramPreprocessor.from_config_dict(cfg.preprocessor)

        self.encoder = ConformerEncoder.from_config_dict(cfg.encoder)

    @property
    def input_types(self):
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal()),
            "length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_length": NeuralType(('B',), LengthsType()),
        }

    def forward(self, input_signal, length):

        processed_signal, processed_length = self.preprocessor(
            input_signal=input_signal, length=length
        )
        
        encoded, encoded_length = self.encoder(
            audio_signal=processed_signal, length=processed_length
        )
        
        return encoded, encoded_length
    def load_preprocessor(self, model_path="/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt", strict=True):
        # This is a bug fix for Accuracy of mel filter bank matrix. Must load this.
        print(f"Loading standalone preprocessor weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.preprocessor.load_state_dict(state_dict, strict=strict)
            print(f"✅ Preprocessor weights loaded successfully (strict={strict})!")
        except FileNotFoundError:
            print(f"⚠️ WARNING: Checkpoint file not found at '{model_path}'. Model is using random weights.")
        except Exception as e:
            print(f"❌ An error occurred during weight loading: {e}")
            import traceback
            traceback.print_exc()
    def load_encoder(self, model_path="/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt", strict=True):
        print(f"Loading standalone encoder weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=strict)
            print(f"✅ Encoder weights loaded successfully (strict={strict})!")
        except FileNotFoundError:
            print(f"⚠️ WARNING: Checkpoint file not found at '{model_path}'. Model is using random weights.")
        except Exception as e:
            print(f"❌ An error occurred during weight loading: {e}")
            import traceback
            traceback.print_exc()
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"实验将在设备上运行: {device}")


    # --- 2. 实例化我们自己的 NeMo 风格的 Encoder ---
    print("\n--- [Step 2] 实例化我们自己的 SemanticEncoder ---")
    my_semantic_encoder = SemanticEncoder(cfg_path="/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder/conf/1b_encoder.yaml").to(device)
    my_semantic_encoder.load_preprocessor("/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt")
    my_semantic_encoder.load_encoder("/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt")
    my_semantic_encoder.eval()
    print("自定义 Encoder 实例化完毕。")

    # --- 3. 从官方模型加载权重到我们的模块中 ---
    print("\n--- [Step 3] 从官方模型加载权重 ---")
    print("正在加载 nvidia/canary-1b-flash...")
    official_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash').to(device)
    official_model.eval()

    # my_semantic_encoder.preprocessor.load_state_dict(official_model.preprocessor.state_dict())
    # my_semantic_encoder.encoder.load_state_dict(official_model.encoder.state_dict())
    # --- 4. 准备输入数据 (使用 24kHz，模块会自动重采样) ---
    print("\n--- [Step 4] 准备 24kHz 输入数据 ---")
    signals_24k = torch.randn(2, 16000 * 4, device=device)
    audio_lengths_24k = torch.tensor([16000 * 4, 16000 * 3], device=device)
    print(f"  - 形状: {signals_24k.shape}, 长度: {audio_lengths_24k.tolist()}")
    
    # --- 5. 对照实验 ---
    print("\n--- [Step 5] 执行对照实验 ---")
    with torch.no_grad():
        # 方法一：调用官方模型的 preprocessor + encoder
        p_signal, p_len = official_model.preprocessor(input_signal=signals_24k, length=audio_lengths_24k)
        official_encoded, official_len = official_model.encoder(audio_signal=p_signal, length=p_len)

        # 方法二：调用我们自己封装的模块
        my_encoded, my_len = my_semantic_encoder(input_signal=signals_24k, length=audio_lengths_24k)
        
    print("两个流程均已完成。")

    # --- 6. 比较结果 ---
    print("\n--- [Step 6] 比较结果 ---")
    shapes_match = (official_encoded.shape == my_encoded.shape)
    print(f"1. 输出形状是否一致? \t{'✅ 是' if shapes_match else '❌ 否'}")
    
    lengths_match = torch.equal(official_len, my_len)
    print(f"2. 输出长度是否一致? \t{'✅ 是' if lengths_match else '❌ 否'}")

    if shapes_match and lengths_match:
        # 此时的误差应该为0或接近机器精度
        tolerance = 1e-7
        values_are_close = torch.allclose(official_encoded, my_encoded, atol=tolerance)
        print(f"3. 数值是否完全一致 (atol={tolerance})? \t{'✅ 是' if values_are_close else '❌ 否'}")
        if not values_are_close:
            abs_diff = torch.abs(official_encoded - my_encoded)
            max_diff = torch.max(abs_diff).item()
            print(f"   - 最大绝对差值: {max_diff:.8f}")

    print("\n--- 对照实验完成 ---")