from typing import List

import numpy as np
from torch import ones_like, Tensor
from torch.utils.data import Dataset
from transformers import AutoProcessor


class MusicGenMelodyDataset(Dataset):
    """Custom dataset for MusicGen Melody finetuning"""

    def __init__(
            self,
            dataset: Dataset,
            processor: AutoProcessor.from_pretrained,
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
        for i in range(len(dataset)):
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
