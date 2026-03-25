import multiprocessing
import os
from os import mkdir
from pathlib import Path

from datasets import load_from_disk

from model.MTS2.train import train

# ---------- 用户手动设置部分 ----------
# 使用 Path 对象，支持 Linux 和 Windows 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_dir, "output/Model/dataset")
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"                      # 验证集文件名
OUTPUT_DIR = os.path.join(project_dir, "output/Model/mts2")       # 模型保存目录

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # ---------- 加载并转换数据集 ----------
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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