import io
import os
import subprocess
import sys

import torch
from torch.utils.data import IterableDataset, get_worker_info
import pretty_midi
import numpy as np
from typing import Iterator, Tuple, List, Optional, Dict, Any
import random
import logging

from data.loader import load_piast_dataset
from utils.mid_preprocessor import midi_to_audio_tensor
from utils.redirector import SuppressOutput

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIDISegmentDataset(IterableDataset):
    """接受包含midi_path字段的Dataset作为输入的数据集"""

    def __init__(
            self,
            midi_dataset,
            batch: int = 32,
            length_seconds: int = 8,
            sample_rate: int = 24000,
            overlap_ratio: float = 0.3,
            shuffle: bool = True,
            infinite: bool = True,
            min_notes_per_segment: int = 3,
            random_start: bool = True,
            max_attempts_per_file: int = 10
    ):
        """
        初始化MIDI切片数据集

        Args:
            midi_dataset: 包含midi_path字段的Dataset
            batch: 批次大小
            length_seconds: 每个片段的长度（秒）
            sample_rate: 音频采样率
            overlap_ratio: 切片重叠比例 (0-1)
            shuffle: 是否打乱数据
            infinite: 是否无限循环
            min_notes_per_segment: 每个片段最少音符数
            random_start: 是否随机选择起始时间
            max_attempts_per_file: 每个文件的最大尝试次数
        """
        self.midi_dataset = midi_dataset
        self.batch = batch
        self.length_seconds = length_seconds
        self.sample_rate = sample_rate
        self.overlap_ratio = max(0.0, min(1.0, overlap_ratio))
        self.shuffle = shuffle
        self.infinite = infinite
        self.min_notes_per_segment = min_notes_per_segment
        self.random_start = random_start
        self.max_attempts_per_file = max_attempts_per_file

        # 计算步长
        self.step_seconds = length_seconds * (1 - overlap_ratio)
        # 获取数据集的长度
        self.dataset_size = len(midi_dataset) if hasattr(midi_dataset, '__len__') else None
        logger.info(f"数据集大小: {self.dataset_size if self.dataset_size is not None else '未知'}")

    def _get_midi_path(self, sample) -> str:
        """
        从样本中提取MIDI文件路径

        Args:
            sample: 数据集样本

        Returns:
            MIDI文件路径字符串
        """
        # 支持多种样本格式
        if isinstance(sample, dict):
            return sample['midi_path']
        elif hasattr(sample, 'midi_path'):
            return sample.midi_path
        elif isinstance(sample, (str, bytes)):
            # 如果样本本身就是路径
            return sample
        else:
            raise ValueError(f"无法从样本中提取midi_path: {type(sample)}")

    def _load_midi_file(self, midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
        """
        加载MIDI文件

        Args:
            midi_path: MIDI文件路径

        Returns:
            pretty_midi.PrettyMIDI对象或None
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            return midi
        except Exception as e:
            logger.debug(f"加载MIDI文件失败 {midi_path}: {e}")
            return None

    def _count_notes_in_segment(self, midi: pretty_midi.PrettyMIDI,
                                start_time: float, end_time: float) -> int:
        """计算片段中的音符数量"""
        note_count = 0
        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.start < end_time and note.end > start_time:
                    note_count += 1
        return note_count

    def _generate_all_segments_for_file(self, sample_idx: int) -> Iterator[Tuple[torch.Tensor, pretty_midi.PrettyMIDI]]:
        """
        为指定样本生成所有片段

        Args:
            sample_idx: 样本索引

        Yields:
            (音频张量, MIDI片段) 元组
        """
        try:
            # 获取样本
            sample = self.midi_dataset[sample_idx]
            midi_path = self._get_midi_path(sample)

            # 加载MIDI文件
            midi = self._load_midi_file(midi_path)
            if midi is None or len(midi.instruments) == 0:
                return

            total_duration = midi.get_end_time()

            # 计算所有可能的片段起始时间
            segment_offsets = self._calculate_segment_offsets(total_duration)

            # 生成所有片段
            for start_time in segment_offsets:
                end_time = min(start_time + self.length_seconds, total_duration)

                # 检查是否有足够音符
                if self._count_notes_in_segment(midi, start_time, end_time) >= self.min_notes_per_segment:
                    # 提取MIDI片段，确保长度为指定长度
                    segment_midi = extract_segment(midi, start_time, end_time)

                    # 合成音频
                    audio, _ = midi_to_audio_tensor(segment_midi, sr = self.sample_rate, duration=self.length_seconds)

                    yield audio, segment_midi

        except Exception as e:
            logger.debug(f"为样本 {sample_idx} 生成片段失败: {e}")
            return

    def _calculate_segment_offsets(self, total_duration: float) -> List[float]:
        """
        计算所有可能的片段起始时间

        Args:
            total_duration: MIDI文件总时长

        Returns:
            起始时间列表
        """
        segment_offsets = []

        if total_duration <= self.length_seconds:
            # 文件太短，只有一个片段
            return [0.0]

        if self.random_start:
            # 随机生成起始时间
            max_start = max(0, total_duration - self.length_seconds)
            # 计算最多能生成多少个片段
            max_segments = int(total_duration / self.step_seconds) + 1

            for i in range(max_segments):
                # 生成覆盖整个文件的起始时间
                if self.overlap_ratio > 0:
                    start_time = min(i * self.step_seconds, max_start)
                else:
                    start_time = min(i * self.length_seconds, max_start)

                segment_offsets.append(start_time)
        else:
            # 按步长生成所有起始时间
            start_time = 0.0
            while start_time + self.length_seconds <= total_duration:
                segment_offsets.append(start_time)
                start_time += self.step_seconds

            # 添加最后一个可能不足length_seconds的片段
            if start_time < total_duration:
                last_start = max(0, total_duration - self.length_seconds)
                segment_offsets.append(last_start)

        # 如果需要打乱，打乱起始时间顺序
        if self.shuffle:
            random.shuffle(segment_offsets)

        return segment_offsets

    def _worker_iter(self, worker_id: int, num_workers: int) -> Iterator[Tuple[torch.Tensor, pretty_midi.PrettyMIDI]]:
        """
        单个worker的迭代器，使用独立的生成片段功能

        Args:
            worker_id: worker ID
            num_workers: worker总数

        Returns:
            数据迭代器
        """
        # 确定worker处理的样本范围
        if self.dataset_size is None:
            # 如果不知道数据集大小，使用迭代器方式
            indices = list(range(1000))
        else:
            # 为每个worker分配数据子集
            samples_per_worker = self.dataset_size // num_workers
            start_idx = worker_id * samples_per_worker

            if worker_id < num_workers - 1:
                end_idx = start_idx + samples_per_worker
            else:
                end_idx = self.dataset_size

            indices = list(range(start_idx, end_idx))

        logger.debug(f"Worker {worker_id}: 处理 {len(indices)} 个样本")

        # 主循环
        while True:
            # 如果需要打乱，打乱样本顺序
            if self.shuffle:
                random.shuffle(indices)

            # 遍历样本（文件），为每个文件生成所有片段
            for sample_idx in indices:
                for audio, segment_midi in self._generate_all_segments_for_file(sample_idx):
                    yield audio, segment_midi

            # 如果不无限循环，则退出
            if not self.infinite:
                break
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, pretty_midi.PrettyMIDI]]:
        """迭代器实现"""
        worker_info = get_worker_info()

        if worker_info is None:
            # 单worker情况
            worker_id = 0
            num_workers = 1
        else:
            # 多worker情况
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # 返回worker的迭代器
        return self._worker_iter(worker_id, num_workers)

    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            "dataset_size": self.dataset_size,
            "segment_length_seconds": self.length_seconds,
            "sample_rate": self.sample_rate,
            "overlap_ratio": self.overlap_ratio,
            "step_seconds": self.step_seconds
        }

def extract_segment(midi: pretty_midi.PrettyMIDI,
                    start_time: float, end_time: float) -> pretty_midi.PrettyMIDI:
    """提取MIDI片段"""
    segment_midi = pretty_midi.PrettyMIDI()

    # 复制元数据
    segment_midi.key_signature_changes = midi.key_signature_changes
    segment_midi.time_signature_changes = midi.time_signature_changes

    # 提取每个乐器的音符
    for instrument in midi.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )

        for note in instrument.notes:
            # 检查音符是否在时间段内
            note_start = note.start
            note_end = note.end

            overlap_start = max(note_start, start_time)
            overlap_end = min(note_end, end_time)

            if overlap_start < overlap_end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity_logits,
                    pitch=note.pitch,
                    start=overlap_start - start_time,
                    end=overlap_end - start_time
                )
                new_instrument.notes.append(new_note)

        if new_instrument.notes:
            segment_midi.instruments.append(new_instrument)

    return segment_midi
def collate_fn(batch):
    """
    自定义collate函数，处理包含pretty_midi.PrettyMIDI对象的批次

    Args:
        batch: 包含(音频张量, MIDI对象)的列表

    Returns:
        批处理后的(音频批次, MIDI列表)
    """
    # 解压批次
    if batch and len(batch[0]) == 2:
        audio_list, midi_list = zip(*batch)

        # 堆叠音频张量
        audio_batch = torch.stack(audio_list, dim=0)

        # MIDI对象保持为列表
        midi_batch = list(midi_list)

        return audio_batch, midi_batch
    else:
        # 如果批次为空或格式不正确，返回空值
        return torch.empty(0), []


# 使用示例
if __name__ == "__main__":
    import soundfile as sf
    # 示例1: 自定义Dataset\

    null_handler = logging.NullHandler()

    logger = logging.getLogger()

    logger.addHandler(null_handler)

    custom_dataset = load_piast_dataset("../../data/dataset/PIAST/")

    # 创建MIDI切片数据集
    midi_dataset = MIDISegmentDataset(
        midi_dataset=custom_dataset["piast-yt"],
        batch=16,
        length_seconds=8,
        sample_rate=24000,
        overlap_ratio=0.3,
        shuffle=True,
        infinite=True,
        min_notes_per_segment=3,
        random_start=True
    )

    # 获取统计信息
    stats = midi_dataset.get_stats()
    print("数据集统计:", stats)

    # 创建数据加载器
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        midi_dataset,
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print("\n测试数据加载...")
    for batch_idx, (audio_batch, midi_batch) in enumerate(dataloader):
        print(f"批次 {batch_idx}: 音频形状: {audio_batch.shape}")
        if batch_idx >= 1:
            break
    '''import logging

    # 打印所有logger
    for name in logging.Logger.manager.loggerDict:
        print(name)'''

    collected_samples = []

    for batch_idx, (audio_batch, midi_batch) in enumerate(dataloader):
        print(f"批次 {batch_idx}: 获得 {len(audio_batch)} 个样本")

        # 收集样本
        for i in range(len(audio_batch)):
            collected_samples.append((audio_batch[i], midi_batch[i]))

        if len(collected_samples) >= 2:
            break

    # 导出前5个样本
    print(f"\n导出 {len(collected_samples)} 个样本...")

    # 创建输出目录
    output_dir = "debug_output/output_samples"
    os.makedirs(output_dir, exist_ok=True)

    # 导出每个样本的音频和MIDI
    for i, (audio, midi) in enumerate(collected_samples):
        # 导出音频
        audio_path = os.path.join(output_dir, f"sample_{i}.wav")
        audio_np = audio.numpy()
        sf.write(audio_path, audio_np, 24000)

        # 导出MIDI
        midi_path = os.path.join(output_dir, f"sample_{i}.mid")
        midi.write(midi_path)

        print(f"样本 {i}:")
        print(f"  音频已保存: {audio_path}")
        print(f"  MIDI已保存: {midi_path}")
        print(f"  音频时长: {len(audio_np) / 24000:.2f}秒")
        print(f"  音频范围: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
        print(f"  MIDI音符数: {sum(len(instr.notes) for instr in midi.instruments)}")


    print("\n测试完成!")