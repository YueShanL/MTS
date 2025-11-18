import json
import os
from pathlib import Path

import pandas as pd
from datasets import DatasetDict
from regex import regex


def load_piast_dataset(repo_path="./dataset/PIAST/", download_if_empty=False):  # --- 加载 piast-at ---
    global Dataset

    try:
        if (not os.path.isdir(repo_path) or len(os.listdir(repo_path)) == 0) and download_if_empty:
            from git import Repo, GitCommandError
            if not os.path.isdir(repo_path): os.makedirs(repo_path)
            print("downloading dataset")
            Repo.clone_from("https://huggingface.co/datasets/Hayeonbang/PIAST", repo_path)
    except Exception as e:
        print(f'unable to download piast dataset to {repo_path} because: {e}')
        return

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


def load_out_dataset(repo_path="../output/Lora/training"):
    """
    加载生成的切片音频文件作为训练数据集

    参数:
        repo_path (str): 包含切片音频文件的目录路径

    返回:
        DatasetDict: 包含训练数据的数据集字典
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(repo_path):
            print(f"输出目录不存在: {repo_path}")
            return None

        # 获取所有音频文件
        audio_files = []
        for ext in ['*.wav']:
            audio_files.extend(list(Path(repo_path).glob(ext)))

        if not audio_files:
            print(f"在 {repo_path} 中未找到音频文件")
            return None

        print(f"找到 {len(audio_files)} 个音频文件")

        # 解析文件名并构建数据集
        dataset_data = {
            "id": [],
            "audio_path": [],
            "text": [],
            "original_name": [],
            "part_index": [],
            "style": []
        }

        # 正则表达式匹配文件名格式: name_partN_style.wav
        pattern_with_part = r'^(.+)_part(\d+)_(.+)\.(wav|mp3|flac)$'
        pattern_without_part = r'^(.+)_(.+)\.(wav|mp3|flac)$'

        for audio_file in audio_files:
            match_with_part = regex.match(pattern_with_part, audio_file.name)
            match_without_part = regex.match(pattern_without_part, audio_file.name)

            if match_with_part:
                # 有part后缀的文件（切片音频）
                original_name = match_with_part.group(1)
                part_index = int(match_with_part.group(2))
                style = match_with_part.group(3)
                file_ext = match_with_part.group(4)

                # 添加到数据集
                dataset_data["id"].append(len(dataset_data["id"]))
                dataset_data["audio_path"].append(str(audio_file))
                dataset_data["text"].append(f"{style} music")
                dataset_data["original_name"].append(original_name)
                dataset_data["part_index"].append(part_index)
                dataset_data["style"].append(style)

            elif match_without_part:
                # 无part后缀的文件（完整长度音频）
                original_name = match_without_part.group(1)
                style = match_without_part.group(2)
                file_ext = match_without_part.group(3)

                # 添加到数据集
                dataset_data["id"].append(len(dataset_data["id"]))
                dataset_data["audio_path"].append(str(audio_file))
                dataset_data["text"].append(f"{style} music")
                dataset_data["original_name"].append(original_name)
                dataset_data["part_index"].append(0)  # 完整音频的part_index设为0
                dataset_data["style"].append(style)
            else:
                print(f"警告: 文件名格式不匹配: {audio_file.name}")

        # 创建数据集
        from datasets import Dataset
        dataset = Dataset.from_dict(dataset_data)

        # 创建 DatasetDict
        from datasets import DatasetDict
        dataset_dict = DatasetDict({"train": dataset})

        print(f"成功加载 {len(dataset)} 个训练样本")

        # 打印一些统计信息
        if len(dataset) > 0:
            styles = set(dataset_data["style"])
            print(f"包含的风格: {styles}")

            # 按原始名称分组统计
            name_counts = {}
            for name in dataset_data["original_name"]:
                name_counts[name] = name_counts.get(name, 0) + 1

            print(f"原始曲目数量: {len(name_counts)}")
            print(f"平均每个曲目的切片数: {len(dataset) / len(name_counts):.2f}")

        return dataset_dict

    except Exception as e:
        print(f"加载输出数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    dataset = load_piast_dataset()
    print(dataset)
    print(dataset['piast-yt'][:10])
