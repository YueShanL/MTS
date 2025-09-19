import ast
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download


def load_piast_dataset(repo_path="./dataset/PIAST/"):  # --- 加载 piast-at ---
    global Dataset
    try:
        # 构建数据集
        dataset_dict = {}

        # 处理 piast-at 部分
        at_path = f'{repo_path}piast_at/'
        if os.path.exists(at_path):
            print("processing piast-at ...")

            # 加载文本数据
            text_df = pd.read_csv(os.path.join(at_path, "at_text.csv"))
            with open(os.path.join(at_path, "at_caption.json"), "r", encoding='UTF-8') as f:
                caption_data = json.load(f)
            # with open(os.path.join(at_path, "tag_list.json"), "r", encoding='UTF-8') as f:
            # tag_data = json.load(f)

            # 加载 MIDI 文件信息
            midi_dir = os.path.join(at_path, "midi")
            midi_files = []
            if os.path.exists(midi_dir):
                midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid') or f.endswith('.midi')]

            # 创建 piast-at 数据集
            at_data = {
                "id": [],
                "text": [],
                # "caption": [],
                "midi_path": []
            }

            # 假设文本数据和MIDI文件有某种对应关系
            # 这里需要根据实际数据结构调整
            for i, row in text_df.iterrows():
                at_data["id"].append(i)
                # at_data["text"].append(row.get("text", ""))

                # 获取对应的caption
                caption = caption_data[i]["caption"].replace(";", ",").split(",")
                at_data["text"].append(caption)

                # 获取对应的tags
                # tags = tag_data.get(str(i), []) if isinstance(tag_data, dict) else []
                # at_data["tags"].append(tags)

                name = row.get("AudioFile", i)
                # 获取对应的MIDI文件路径
                midi_path = os.path.join(midi_dir, f"{name}.mid") if i < len(midi_files) else ""
                at_data["midi_path"].append(midi_path if os.path.exists(midi_path) else "")

            # 创建数据集
            from datasets import Dataset
            at_dataset = Dataset.from_dict(at_data)
            dataset_dict["piast-at"] = at_dataset

        # 处理 piast-yt 部分
        yt_path = os.path.join(repo_path, "piast_yt")
        if os.path.exists(yt_path):
            print("processing piast-yt...")

            # 加载文本数据
            with open(os.path.join(yt_path, "youtube.json"), "r", encoding='UTF-8') as f:
                yt_data = json.load(f)

            # 加载 MIDI 文件信息
            midi_dir = os.path.join(yt_path, "midi")
            midi_files = []
            if os.path.exists(midi_dir):
                midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid') or f.endswith('.midi')]

            # 创建 piast-yt 数据集
            yt_dataset_data = {
                "id": [],
                "text": [],
                "midi_path": []
            }

            for i, item in enumerate(yt_data):
                midi_path = os.path.join(midi_dir, f"{item['track_id']}.mid") if i < len(midi_files) else ""

                if midi_path == "" or not os.path.exists(midi_path):
                    continue
                yt_dataset_data["id"].append(i)
                yt_dataset_data["text"].append(item['tag'][0].split(","))
                yt_dataset_data["midi_path"].append(midi_path)

            # 创建数据集
            yt_dataset = Dataset.from_dict(yt_dataset_data)
            dataset_dict["piast-yt"] = yt_dataset

        return DatasetDict(dataset_dict)

    except Exception as e:
        print(f"loading failed: {e}")
        return None


if __name__ == "__main__":
    dataset = load_piast_dataset()
    print(dataset)
    print(dataset['piast-at'][:10])
    print(dataset['piast-yt'][:10])
