import multiprocessing
import os

from datasets import load_dataset, Dataset

from model.MTS2.module import MTSGen2
from model.MTS2.rl_train import train

# ---------- 用户手动设置部分 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
#DATA_DIR = os.path.join(project_dir, "output/Model/dataset")
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"                      # 验证集文件名
OUTPUT_DIR = os.path.join(project_dir, "output/Model/mts2_rl")       # 模型保存目录
MODEL_DIR = os.path.join(project_dir, "output/Model/mts2/best")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # ---------- 加载并转换数据集 ----------
    # 确保目录存在
    #os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #dataset = load_from_disk(str(DATA_DIR))
    seed = 43
    val_size = 6
    train_size = 20

    #dataset = dataset.train_test_split(test_size=0.1,seed=42,shuffle=True)
    model = MTSGen2.from_pretrained(MODEL_DIR)
    train_dataset = load_dataset("astune/mts_rl_dataset", streaming=True, split='train').skip(val_size).take(train_size)
    eval_dataset = load_dataset("astune/mts_rl_dataset", streaming=True, split='train').take(val_size)


    # ---------- 开始训练 ----------
    train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(OUTPUT_DIR),
        # model_config=custom_config,           # 若需自定义配置，取消注释
        # training_args=custom_training_args,   # 若需自定义训练参数，取消注释
        datasize=train_size,
        freeze_basic_pitch=True,
        seed=42,
        model=model,
    )