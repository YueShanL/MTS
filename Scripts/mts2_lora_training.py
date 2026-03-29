import os
from logging import Logger

import numpy as np
import torch
import torchaudio
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.functional import Tensor
from transformers import TrainingArguments

from model.MTS2.module import MTSGen2
from model.MTS2.train import train
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.trainer import train_mixed_model

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

def resample_audio(example):
    """
    将音频从原始采样率重采样到 22056 Hz。
    """
    waveform = Tensor(example['audio_input'])
    orig_sr = 32000
    target_sr = 22050

    # 如果已经是目标采样率，直接返回
    if orig_sr == target_sr:
        return example

    waveform = waveform.unsqueeze(0)   # (1, samples)

    # 重采样（使用 functional.resample 避免创建重复的转换器）
    resampled_waveform = nn.functional.normalize(torchaudio.functional.resample(waveform, orig_sr, target_sr), dim = 1)

    resampled_array = resampled_waveform.squeeze(0).tolist()

    example['audio_input'] = resampled_array
    return example

if __name__ == '__main__':
    output_path = os.path.join(project_dir, "output/Model/Lora/mts2")
    pretrained_dir = os.path.join(project_dir, "output/Model/mts2/best")
    logger = Logger(__name__)
    device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

    dataset = Dataset.load_from_disk('mts_lora_dataset').rename_column('audio_inputs', 'audio_input').map(
        resample_audio, num_proc=4)

    model = MTSGen2.from_pretrained(pretrained_dir)
    model.to('cuda')
    target_modules = (
            [
                "enc_to_dec_proj",
                "audio_enc_to_dec_proj",
                "k_proj",
                "v_proj",
                "q_proj",
                "out_proj",
                "fc1",
                "fc2",
                "lm_heads.0",
            ]
        )

    training_args = TrainingArguments(
        output_dir=output_path,
        logging_dir=os.path.join(output_path, "runs"),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-3,
        logging_steps=2,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,

        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="lora_only",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    logger.info(f"Modules with Lora: {model.targeted_module_names}")

    train(
        train_dataset=dataset,
        eval_dataset=None,
        output_dir=str(output_path),
        # model_config=custom_config,           # 若需自定义配置，取消注释
        training_args=training_args,   # 若需自定义训练参数，取消注释
        freeze_basic_pitch=True,
        seed=42,
    )

    model.save_pretrained(str(output_path))