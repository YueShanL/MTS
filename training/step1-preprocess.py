import os

import torch
from openunmix import predict
import torchaudio
from audiocraft.data.audio import audio_write
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from Scripts.generate_lora_data import linux
from data.loader import load_piast_dataset, load_out_dataset

testSource = "./asset/"
filename = 'songTest.mp3'
'umxhq'

melody, sr = torchaudio.load(testSource + filename)


def prepare_musicgen_dataset(param, processor):
    pass


def setup_training():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody-large")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody-large")

    # 加载数据集
    dataset = create_training_dataset()

    if dataset is None:
        print("无法加载训练数据集")
        return

    # 准备数据集
    prepared_dataset = prepare_musicgen_dataset(dataset["train"], processor)

    # 创建数据加载器
    dataloader = DataLoader(
        prepared_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn  # 使用之前定义的collate_fn
    )

    return model, processor, dataloader

def create_training_dataset(piast_path="./dataset/PIAST/",
                            output_path="./output/Lora/training",
                            download_piast=False):
    """
    创建用于训练的数据集，结合原始PIAT数据和生成的音频数据

    参数:
        piast_path (str): PIAST数据集路径
        output_path (str): 生成的音频文件路径
        download_piast (bool): 是否下载PIAST数据集（如果不存在）

    返回:
        DatasetDict: 合并后的训练数据集
    """
    # 加载原始PIAST数据集
    piast_dataset = load_piast_dataset(piast_path, download_if_empty=download_piast)

    # 加载生成的音频数据集
    output_dataset = load_out_dataset(output_path)

    # 如果两个数据集都成功加载，合并它们
    if piast_dataset is not None and output_dataset is not None:
        # 注意：这里需要确保两个数据集有相同的特征
        # 我们可以选择只使用其中一个，或者合并它们

        print("原始PIAST数据集和生成的音频数据集都已加载")
        print("PIAST数据集大小:", len(piast_dataset.get("piast-at", [])) + len(piast_dataset.get("piast-yt", [])))
        print("生成的音频数据集大小:", len(output_dataset["train"]))

        # 只使用生成的音频数据
        training_dataset = output_dataset

        # 合并两个数据集（需要调整特征对齐）
        # training_dataset = merge_datasets(piast_dataset, output_dataset)

        return training_dataset

    elif output_dataset is not None:
        print("使用生成的音频数据集进行训练")
        return output_dataset

    elif piast_dataset is not None:
        print("使用原始PIAST数据集进行训练")
        # 将PIAST数据集转换为训练格式
        return convert_piast_to_training_format(piast_dataset)

    else:
        print("无法加载任何数据集")
        return None


def convert_piast_to_training_format(piast_dataset):
    """
    将PIAST数据集转换为训练格式

    参数:
        piast_dataset: 原始PIAST数据集

    返回:
        DatasetDict: 转换为训练格式的数据集
    """
    try:
        from datasets import Dataset, DatasetDict
        import torchaudio

        training_data = {
            "id": [],
            "audio_path": [],
            "text": [],
            "original_name": [],
            "part_index": [],
            "style": []
        }

        # 处理piast-at部分
        if "piast-at" in piast_dataset:
            at_dataset = piast_dataset["piast-at"]
            for i, item in enumerate(at_dataset):
                # 如果有MIDI文件，可以转换为音频
                midi_path = item["midi_path"]
                if midi_path and os.path.exists(midi_path):
                    # 这里可以添加MIDI到音频的转换逻辑
                    # 暂时使用MIDI路径作为占位符
                    training_data["id"].append(len(training_data["id"]))
                    training_data["audio_path"].append(midi_path)  # 实际使用时需要转换为音频
                    training_data["text"].append(
                        " ".join(item["text"]) if isinstance(item["text"], list) else item["text"])
                    training_data["original_name"].append(f"piast-at_{i}")
                    training_data["part_index"].append(0)  # 原始文件没有分片
                    training_data["style"].append("unknown")  # 从文本中提取风格

        # 处理piast-yt部分
        if "piast-yt" in piast_dataset:
            yt_dataset = piast_dataset["piast-yt"]
            for i, item in enumerate(yt_dataset):
                midi_path = item["midi_path"]
                if midi_path and os.path.exists(midi_path):
                    training_data["id"].append(len(training_data["id"]))
                    training_data["audio_path"].append(midi_path)  # 实际使用时需要转换为音频
                    training_data["text"].append(
                        " ".join(item["text"]) if isinstance(item["text"], list) else item["text"])
                    training_data["original_name"].append(f"piast-yt_{i}")
                    training_data["part_index"].append(0)
                    training_data["style"].append("unknown")

        dataset = Dataset.from_dict(training_data)
        return DatasetDict({"train": dataset})

    except Exception as e:
        print(f"转换PIAST数据集失败: {e}")
        return None


def separate_song(audio, sr, export: str = None, target=None, modelName="umxl"):
    separated = predict.separate(audio[None], rate=sr, model_str_or_path=modelName, targets=target)
    if export is not None:
        for name, tensor in zip(separated.keys(), separated.values()):
            # Will save under {modelName}.{export}.{name}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{modelName}.{export}.{name}', tensor[0].cpu(), sr, strategy="loudness",
                        loudness_compressor=True)
    return separated


# alternative methods
def vocal_to_note(audio, export: str = None, ):
    return


def combine_melody():
    return


def slice():
    return


def speedShift():
    return
