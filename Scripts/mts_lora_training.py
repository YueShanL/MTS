from logging import Logger

import numpy as np
import torch
import torchaudio
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.functional import Tensor

from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.trainer import train_mixed_model

def resample_audio(example):
    """
    将音频从原始采样率重采样到 24000 Hz。
    """
    waveform = Tensor(example['audio_input'])
    orig_sr = 32000
    target_sr = 24000

    # 如果已经是目标采样率，直接返回
    if orig_sr == target_sr:
        return example

    waveform = waveform.unsqueeze(0)   # (1, samples)

    # 重采样（使用 functional.resample 避免创建重复的转换器）
    resampled_waveform = nn.functional.normalize(torchaudio.functional.resample(waveform, orig_sr, target_sr), dim = 1)

    resampled_array = resampled_waveform.squeeze(0).tolist()

    # 更新样本
    example['audio_input'] = resampled_array
    return example


linux = 0
debug = 0
if __name__ == '__main__':
    output_path = "output/Model/tf_50m" if linux else "../output/Model/Lora"
    logger = Logger(__name__)
    device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

    dataset = Dataset.load_from_disk('mts_lora_dataset').rename_column('audio_inputs', 'audio_input').map(resample_audio, num_proc=4)

    config = MTSGenConfig.mtsGen_150m()
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch59.pth'))
    model.to('cuda')

    target_modules = [
        # 音频投影
        "audio_projection",

        # 音符嵌入模块（NoteEmbedding）
        "note_embedding.duration_embedding",
        "note_embedding.fret_embedding",
        "note_embedding.technique_embedding",
        "note_embedding.string_attention.out_proj",   # 注意：仅out_proj是独立的Linear
        "note_embedding.ffn.0",                        # ffn中第一个线性层
        "note_embedding.ffn.3",                        # ffn中第二个线性层
        "note_embedding.feature_fusion.0",             # feature_fusion第一个线性层
        "note_embedding.feature_fusion.4",             # feature_fusion第二个线性层

        # 融合编码器（每层）
        "fusion_encoder.layers.*.self_attn.out_proj",
        "fusion_encoder.layers.*.linear1",
        "fusion_encoder.layers.*.linear2",

        # 自回归解码器（每层）
        "autoregressive_decoder.layers.*.self_attn.out_proj",
        "autoregressive_decoder.layers.*.multihead_attn.out_proj",
        "autoregressive_decoder.layers.*.linear1",
        "autoregressive_decoder.layers.*.linear2",

        # 输出头
        "duration_head",
        # 品位头（每根弦一个线性层）
        *[f"fret_heads.{i}" for i in range(config.num_strings)],
        # 技巧头（每根弦一个线性层）
        *[f"technique_heads.{i}" for i in range(config.num_strings)],
    ]

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    logger.info(f"Modules with Lora: {model.targeted_module_names}")
    try:
        train_mixed_model(model, dataset, val_dataset=None,
                                  num_epochs=10, batch_size=4, output_path=output_path, scheduler_type = "teacher_forced")
    except Exception as e:
        logger.error(e)
    model.save_pretrained(output_path)