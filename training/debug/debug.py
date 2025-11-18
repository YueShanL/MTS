"""
最小可运行示例 - MusicGen Melody 训练
这个脚本可以直接运行，用于验证环境和理解数据流
"""

import torch
import numpy as np
from transformers import (
    AutoProcessor,
    MusicgenMelodyForConditionalGeneration,
    EncodecModel
)


def test_audio_encoder():
    """测试 audio_encoder 的正确使用方法"""
    print("测试 Audio Encoder...")

    # 加载 EnCodec
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    audio_encoder.eval()

    # 创建测试音频（正弦波）
    sample_rate = 32000
    duration = 2  # 秒
    freq = 440  # Hz
    t = np.linspace(0, duration, duration * sample_rate)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # 转换为张量
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # [1, length]
    print(f"原始音频形状: {audio_tensor.shape}")

    # 添加通道维度（EnCodec 需要这个）
    audio_tensor = audio_tensor.unsqueeze(1)  # [1, 1, length]
    print(f"添加通道后: {audio_tensor.shape}")

    # 编码
    with torch.no_grad():
        encoded = audio_encoder.encode(audio_tensor)
        codes = encoded.audio_codes.squeeze(0)
        print(f"编码后 codes 形状: {codes.shape}")
        print(f"Codes 数据类型: {codes.dtype}")

    return codes


def test_musicgen_forward_pass():
    """测试 MusicGen 的完整前向传播"""
    print("\n测试 MusicGen Forward Pass...")

    # 加载模型和处理器
    model_name = "facebook/musicgen-melody"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    # 加载 audio encoder（用于准备标签）
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    audio_encoder.eval()

    # 准备数据
    text = ["upbeat electronic dance music"]

    # 创建目标音频
    sample_rate = 32000
    duration = 3
    audio = np.random.randn(duration * sample_rate).astype(np.float32) * 0.1

    # 处理文本
    text_inputs = processor.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True
    )
    print(f"文本输入形状 - input_ids: {text_inputs['input_ids'].shape}")

    # 处理旋律音频（用作条件）- 使用 feature_extractor
    melody_features = processor.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    print(f"旋律特征形状: {melody_features['input_features'].shape}")

    # 准备目标音频的 codes
    target_audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(1)  # [1, 1, length]
    with torch.no_grad():
        encoded = audio_encoder.encode(target_audio_tensor)
        target_codes = encoded.audio_codes.squeeze(0).transpose(1, 2)
    print(f"目标 codes 形状: {target_codes.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            input_features=melody_features['input_features'],  # 旋律特征
            labels=target_codes  # 离散 codes 作为标签
        )
        print(f"损失: {outputs.loss.item():.4f}")

    return outputs


def simple_training_loop():
    """简单的训练循环示例"""
    print("\n运行简单训练循环...")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(
        "facebook/musicgen-melody",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
    audio_encoder.eval()

    # 应用 LoRA（可选）
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=4,  # 很小的 rank 用于测试
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"]
        )

        model = get_peft_model(model, lora_config)
        print("LoRA 已应用")
        model.print_trainable_parameters()
    except ImportError:
        print("PEFT 未安装，跳过 LoRA")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 训练几步
    model.train()
    for step in range(3):
        # 创建假数据
        text = f"music description {step}"
        audio = np.random.randn(32000 * 2).astype(np.float32) * 0.1

        # 处理输入
        text_inputs = processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )

        # 编码目标音频
        audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            target_codes = audio_encoder.encode(audio_tensor).audio_codes

        # 前向传播
        outputs = model(
            input_ids=text_inputs['input_ids'].to(device),
            attention_mask=text_inputs['attention_mask'].to(device),
            labels=target_codes
        )

        loss = outputs.loss
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("训练循环完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("MusicGen Melody 最小示例")
    print("=" * 60)

    # 1. 测试 audio encoder
    codes = test_audio_encoder()

    # 2. 测试完整前向传播
    outputs = test_musicgen_forward_pass()

    # 3. 运行简单训练
    simple_training_loop()

    print("\n✅ 所有测试通过！")
    print("\n关键要点回顾:")
    print("1. audio_encoder 需要形状 [batch, channels, length] 的音频")
    print("2. channels 必须是 1 或 2")
    print("3. input_features 是 mel-spectrogram，不是原始音频")
    print("4. labels 必须是离散的 codes (Long 类型)")


if __name__ == "__main__":
    main()
