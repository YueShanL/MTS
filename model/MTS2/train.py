import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import Trainer, TrainingArguments

from model.MTS2.data import DataCollatorForMTSGen2WithWaveform
from model.MTS2.module import MTSGen2Config, MTSGen2

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    freeze_basic_pitch: bool = field(default=True)  # 通常冻结 BasicPitch

@dataclass
class DataArguments:
    data_dir: str = field(default="./data")
    train_split: str = field(default="train")
    eval_split: str = field(default="eval")
    dataset_format: str = field(default="json")

def train(
    train_dataset,
    eval_dataset,
    output_dir: str,
    model_config: Optional[MTSGen2Config] = None,
    training_args: Optional[TrainingArguments] = None,
    freeze_basic_pitch: bool = True,
    seed: int = 42,
):
    """
    训练 MTSGen2 模型。

    Args:
        train_dataset: Hugging Face Dataset 对象，包含 'waveform', 'input_ids', 'labels' 字段
        eval_dataset: 验证集，同上
        output_dir: 模型保存目录
        model_config: 模型配置，若为 None 则使用默认配置
        training_args: 训练参数，若为 None 则使用默认参数
        freeze_basic_pitch: 是否冻结 BasicPitch 参数
        seed: 随机种子
    """

    # 默认配置
    if model_config is None:
        model_config = MTSGen2Config()

    # 默认训练参数
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "runs"),
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            learning_rate=1e-4,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            seed=seed,
            remove_unused_columns=False,
            dataloader_num_workers=4
        )

    # 初始化模型
    model = MTSGen2(model_config).to('cuda')

    # 数据整理器
    collator = DataCollatorForMTSGen2WithWaveform(
        pad_token_id=model_config.decoder_start_token_id,
        label_pad_token_id=-100
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=None,
    )

    # 训练
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"模型已保存至 {output_dir}")