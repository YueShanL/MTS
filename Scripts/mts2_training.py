import multiprocessing
import os
from pathlib import Path

import multiprocess
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments

from model.MTS2.data import convert_example
from model.MTS2.train import train

# ---------- 用户手动设置部分 ----------
# 使用 Path 对象，支持 Linux 和 Windows 路径
DATA_DIR = Path("./../output/Model/dataset")                # 数据目录，相对路径或绝对路径均可
TRAIN_SPLIT = "train"                    # 训练集文件名（不含扩展名）
EVAL_SPLIT = "eval"                      # 验证集文件名
OUTPUT_DIR = Path("./../output/Model/mts2")           # 模型保存目录

# 可选：自定义模型配置
# from mtsgen2_model import MTSGen2Config
# custom_config = MTSGen2Config(...)

# 可选：自定义训练参数
# custom_training_args = TrainingArguments(
#     output_dir=str(OUTPUT_DIR),
#     num_train_epochs=20,
#     per_device_train_batch_size=2,
#     ...
# )
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # ---------- 加载并转换数据集 ----------
    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据集（load_dataset 接受字符串路径，所以转换为 str）
    dataset = load_from_disk(str(DATA_DIR))

    dataset = dataset.train_test_split(test_size=0.1,seed=42,shuffle=True)
    train_dataset = dataset["train"].remove_columns(["context_notes"])
    eval_dataset = dataset["test"].remove_columns(["context_notes"])


    # ---------- 开始训练 ----------
    train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(OUTPUT_DIR),
        # model_config=custom_config,           # 若需自定义配置，取消注释
        # training_args=custom_training_args,   # 若需自定义训练参数，取消注释
        freeze_basic_pitch=True,
        seed=42,
    )