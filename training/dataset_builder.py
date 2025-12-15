import os
import re
from collections import defaultdict

import torch
import torchaudio
from pathlib import Path
from datasets import Dataset, DatasetDict
import json
import pandas as pd
from numpy import shape
from torch.utils.data import DataLoader

from transformers import AutoProcessor

from data.loader import load_piast_dataset
from utils.mid_preprocessor import midi_to_audio_tensor


def parse_stylized_filename(filename):
    """
    解析风格化音频文件名，处理包含多个"part"关键词的复杂情况

    参数:
        filename: 音频文件名

    返回:
        tuple: (original_id, part_index, style) 或 (None, None, None) 如果解析失败
    """
    # 移除文件扩展名
    base_name = os.path.splitext(filename)[0]

    # 尝试匹配格式: id_partX_style 或 id_style
    # 处理包含多个"part"关键词的情况

    # 首先尝试匹配包含part的格式
    # 匹配模式: 任意字符 + "_part" + 数字 + "_" + 任意字符 (风格)
    part_pattern = r'^(.+)_part(\d+)_(.+)$'
    match = re.match(part_pattern, base_name)

    if match:
        original_id = match.group(1)
        part_index = int(match.group(2))
        style = match.group(3)
        return original_id, part_index, style

    # 如果没有匹配到part格式，尝试匹配无part的格式
    # 匹配模式: 任意字符 + "_" + 任意字符 (风格)
    no_part_pattern = r'^(.+)_(.+)$'
    match = re.match(no_part_pattern, base_name)

    if match:
        original_id = match.group(1)
        style = match.group(2)
        return original_id, -1, style  # -1 表示没有分片

    # 如果都不匹配，返回失败
    return None, None, None


def load_audio_dataset(piast_data, generated_audio_dir):
    """
    加载风格化音频数据集，处理复杂的文件名结构

    参数:
        piast_data: 从load_piast_dataset函数返回的数据
        generated_audio_dir: 生成的风格化音频目录
        max_audio_length: 最大音频长度（秒）
        sample_rate: 采样率

    返回:
        dataset: 风格化音频数据集
    """
    # 加载生成的音频文件
    generated_audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        generated_audio_files.extend(list(Path(generated_audio_dir).glob(ext)))

    print(f"找到 {len(generated_audio_files)} 个生成的音频文件")

    # 构建数据集数据
    dataset_data = {
        "audio_path": [],  # 音频文件路径
        "text": [],  # 文本描述
        "original_path": [],  # 原始ID
        "part_index": [],  # 分片索引（-1表示未分割）
        "duration": [],  # 音频时长
    }

    # 按ID分组音频文件
    audio_files_by_id = defaultdict(list)

    # 解析生成的音频文件名
    for audio_file in generated_audio_files:
        # 使用新的解析函数
        original_id, part_index, style = parse_stylized_filename(audio_file.name)

        if original_id is None:
            print(f"警告: 无法解析文件名: {audio_file.name}")
            continue

        try:
            info = torchaudio.info(str(audio_file))
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"无法获取音频文件 {audio_file.name} 的时长: {e}")
            duration = 0

        # 按ID和风格分组
        key = original_id
        audio_files_by_id[key].append({
            "style": style,
            "path": str(audio_file),
            "part_index": part_index,
            "duration": duration
        })

    # 为每个音频文件找到对应的文本描述
    for original_id, audio_files in audio_files_by_id.items():
        # 查找对应的文本描述
        text_description = ""

        # 从PIAST数据中查找
        if piast_data is not None:
            for i, pid in enumerate(piast_data["midi_path"]):
                id = os.path.splitext(os.path.basename(pid))[0]
                if id == original_id:
                    text_list = piast_data["text"][i]
                    if isinstance(text_list, list):
                        text_description = ",".join(text_list)
                    else:
                        text_description = str(text_list)
                    break

        text_description += ", piano cover, "

        # 为每个音频文件创建样本
        for audio_file_info in audio_files:
            text = text_description + audio_file_info["style"]
            dataset_data["audio_path"].append(audio_file_info["path"])
            dataset_data["text"].append(text)
            dataset_data["original_path"].append(pid)
            dataset_data["part_index"].append(audio_file_info["part_index"])
            dataset_data["duration"].append(audio_file_info["duration"])

    # 创建数据集
    dataset = Dataset.from_dict(dataset_data)

    print(f"成功加载音频数据集，包含 {len(dataset)} 个样本")

    # 打印统计信息
    if len(dataset) > 0:
        part_counts = defaultdict(int)
        for part_idx in dataset_data["part_index"]:
            if part_idx == -1:
                part_counts["未分割"] += 1
            else:
                part_counts[f"分片{part_idx}"] += 1

        print("音频分割统计:")
        for part_type, count in part_counts.items():
            print(f"  {part_type}: {count} 个样本")

    return dataset


def process_style_transfer_dataset(dataset: Dataset, generator = False, cache_dir="./dataset_cache"):
    """
    处理风格转换数据集，添加音频编码
    """
    time = {}

    def process_example(example, sample_rate=32000):
        # 加载参考音频
        input_waveform, input_sr = torchaudio.load(example["audio_path"])

        try:

            # 转换为单声道
            if input_waveform.shape[0] > 1:
                input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
            else:
                input_waveform = input_waveform.unsqueeze(0)

            # 重采样
            if input_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(input_sr, sample_rate)
                input_waveform = resampler(input_waveform)

            # 加载目标音频
            soundfont_path = "../asset/GeneralUser-GS.sf2"
            target_waveform, target_sr = midi_to_audio_tensor(example["original_path"], soundfont_path=soundfont_path)

            if example['part_index'] > 0:
                print(
                    f'take {int(time[example["original_path"]])} to {int(time[example["original_path"]] + example["duration"] * input_sr)} with length {len(target_waveform)}')
                target_waveform = target_waveform[time[example["original_path"]]: int(
                    time[example["original_path"]] + example["duration"] * input_sr)]
            else:
                time[example["original_path"]] = 0

            target_waveform[None].expand(1, -1, -1)

            time[example["original_path"]] += int(example["duration"] * input_sr)

            # 重采样
            if target_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(target_sr, sample_rate)
                target_waveform = resampler(target_waveform)

            return {
                "input_audio_values": input_waveform.squeeze(),
                "target_audio_values": target_waveform.squeeze(),
                "text": example["text"],
            }

        except Exception as e:
            print(f'failed to process {example} because: {e}')
            return None

    # 应用处理函数
    if generator:
        for example in dataset:
            yield process_example(example)
    else:
        processed_dataset = dataset.sort(column_names='part_index').map(
            process_example,
            remove_columns=dataset.column_names,
            cache_file_name=os.path.join(cache_dir, "style_transfer_cache.arrow"),  # 缓存到磁盘
            writer_batch_size=10,  # 控制写入批次大小
            # load_from_cache_file=False,  # 强制重新处理
        ).filter(lambda example: example is not None)

        return processed_dataset


if __name__ == '__main__':
    piast = load_piast_dataset('../data/dataset/PIAST/')

    data = load_audio_dataset(
        piast_data=piast['piast-yt'],
        generated_audio_dir="../output/Lora/training"
    )

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    data = process_style_transfer_dataset(data, processor)

    print(data.info)
    DatasetDict({"train": data}).save_to_disk("../output/Lora/dataset")
    # for i, name in enumerate(data['audio_path']):
    # if name == '..\\output\\Lora\\training\\1xanU4QRWnI_part_7_part0_Alternative Rock.wav':
    # print(data[i:i + 3])
    # data = process_style_transfer_dataset(Dataset.from_dict(data[i:i + 3]), processor)
    # print(data.info)
