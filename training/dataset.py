from typing import Iterator

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoProcessor


class MusicGenMelodyDataset(Dataset):
    """Custom dataset for MusicGen Melody finetuning"""

    def __init__(
            self,
            dataset: Dataset,
            processor: AutoProcessor.from_pretrained,
            dataset_len: int = 50,
            max_length: int = 30 * 32000,  # ~30 seconds at 32kHz
            sampling_rate: int = 32000,
            sliding_window: int = 10 * 32000
    ):
        self.target_audio_values = dataset['target_audio_values']
        self.input_audio_values = dataset['input_audio_values']
        self.texts = dataset['text']
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.step = max_length - sliding_window

        self.slice_counts = []
        for i in range(dataset_len):
            audio_length = len(dataset['input_audio_values'][i])
            num_slices = max(1, (audio_length - sliding_window) // self.step + 1)
            self.slice_counts.append(num_slices)

        self.cumulative_slices = [0]
        for count in self.slice_counts:
            self.cumulative_slices.append(self.cumulative_slices[-1] + count)

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, idx):
        # Get raw data
        slice_idx = 0
        audio_idx = 0
        for i in range(len(self.cumulative_slices) - 1):
            if self.cumulative_slices[i] <= idx < self.cumulative_slices[i + 1]:
                audio_idx = i
                slice_idx = idx - self.cumulative_slices[i]
                break

        audio_length = len(self.input_audio_values[audio_idx])
        start = slice_idx * self.step
        end = min(start + self.max_length, audio_length)

        input_audio = self.input_audio_values[audio_idx][start:end]
        target_audio = self.target_audio_values[audio_idx][start:end]
        text = self.texts[audio_idx]

        # padding
        if len(input_audio) < self.max_length:
            pad_length = self.max_length - len(input_audio)
            input_audio = np.pad(input_audio, (0, pad_length), mode='constant')
            target_audio = np.pad(target_audio, (0, pad_length), mode='constant')

        return {
            'input_ids': [text],
            'input_features': Tensor(input_audio),  # dim = 1
            'labels': Tensor(target_audio).unsqueeze(0),  # target_inputs['input_features'] dim = 2
            'sample_rate': self.sampling_rate
        }


class MusicGenMelodyIterableDataset(IterableDataset):
    """Custom IterableDataset for MusicGen Melody finetuning"""

    def __init__(
            self,
            dataset: IterableDataset,
            processor: AutoProcessor,
            max_length: int = 30 * 32000,  # ~30 seconds at 32kHz
            sampling_rate: int = 32000,
            sliding_window: int = 10 * 32000,
            shuffle: bool = False,
            buffer_size: int = 10
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.sliding_window = sliding_window
        self.step = max_length - sliding_window
        self.shuffle = shuffle
        self.buffer_size = buffer_size

        # 验证 step 的有效性
        if self.step <= 0:
            raise ValueError("step must be positive, check max_length and sliding_window values")

    def __iter__(self) -> Iterator[dict]:
        """返回数据集的迭代器"""
        # 创建基础迭代器
        if self.shuffle:
            # 对于可迭代数据集，我们使用缓冲区进行shuffle
            dataset_iter = self._shuffled_iterator()
        else:
            dataset_iter = iter(self.dataset)

        # 为每个样本生成切片
        for example in dataset_iter:
            yield from self._generate_slices(example)

    def _shuffled_iterator(self) -> Iterator:
        """创建带shuffle的迭代器"""
        buffer = []

        for example in self.dataset:
            buffer.append(example)

            # 当缓冲区达到指定大小时，进行shuffle并yield
            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    np.random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []

        # 处理缓冲区中剩余的数据
        if buffer:
            if self.shuffle:
                np.random.shuffle(buffer)
            for item in buffer:
                yield item

    def _generate_slices(self, example: dict) -> Iterator[dict]:
        """为单个音频样本生成所有切片"""
        input_audio = example['input_audio_values']
        target_audio = example['target_audio_values']
        text = example['text']

        audio_length = len(input_audio)

        # 跳过空音频
        if audio_length == 0:
            return

        # 计算切片数量
        num_slices = max(1, (audio_length - self.sliding_window) // self.step + 1)

        # 生成所有切片
        for slice_idx in range(num_slices):
            start = slice_idx * self.step
            end = min(start + self.max_length, audio_length)

            # 提取切片
            input_slice = input_audio[start:end]
            target_slice = target_audio[start:end]

            # 填充不足长度的切片
            if len(input_slice) < self.max_length:
                pad_length = self.max_length - len(input_slice)
                input_slice = np.pad(input_slice, (0, pad_length), mode='constant')
            if len(target_slice) < self.max_length:
                pad_length = self.max_length - len(target_slice)
                target_slice = np.pad(target_slice, (0, pad_length), mode='constant')

            yield {
                'input_ids': [text],
                'input_features': Tensor(input_slice),
                'labels': Tensor(target_slice).unsqueeze(0),
                'sample_rate': self.sampling_rate
            }


def create_musicgen_dataset(
        dataset,
        processor: AutoProcessor,
        **kwargs
):
    """根据数据集类型创建适当的MusicGen数据集"""
    if isinstance(dataset, IterableDataset):
        return MusicGenMelodyIterableDataset(dataset, processor, **kwargs)
    else:
        return MusicGenMelodyDataset(dataset, processor, **kwargs)
