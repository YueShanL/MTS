import io
import logging
import os
from typing import Iterator, Dict, Any, Union, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
import pretty_midi
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class HFDataConfig:
    """HuggingFace数据集配置"""
    audio_column_name: str = "audio"
    midi_column_name: str = "midi_bytes"
    segment_info_column_name: str = "segment_info"
    sample_rate: int = 24000
    audio_dtype: str = "float32"
    include_midi_bytes: bool = True
    include_segment_info: bool = True


class MIDISegmentToHFGenerator:
    """
    将MIDISegmentDataset转换为HuggingFace数据集格式的生成器
    """

    def __init__(
            self,
            midi_segment_dataset,
            config: HFDataConfig = None,
            max_samples: Optional[int] = None,
    ):
        """
        初始化转换生成器

        Args:
            midi_segment_dataset: MIDISegmentDataset实例
            config: HuggingFace数据配置
            max_samples: 最大样本数（None表示无限制）
        """
        self.dataset = midi_segment_dataset
        self.config = config or HFDataConfig()
        self.max_samples = max_samples

    @staticmethod
    def _midi_to_bytes(midi_obj) -> bytes:
        """将PrettyMIDI对象转换为字节"""
        with io.BytesIO() as buffer:
            midi_obj.write(buffer)
            return buffer.getvalue()

    @staticmethod
    def _extract_segment_info(midi_obj) -> Dict[str, Any]:
        """从MIDI片段中提取信息"""
        if not midi_obj.instruments:
            return {
                "note_count": 0,
                "instrument_count": 0,
                "is_drum": False,
                "mean_pitch": 0.0,
                "pitch_range": [0, 0],
                "mean_velocity": 0.0,
                "mean_duration": 0.0,
            }

        pitches, velocities, durations = [], [], []
        is_drum = any(instr.is_drum for instr in midi_obj.instruments)

        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                pitches.append(note.pitch)
                velocities.append(note.velocity)
                durations.append(note.end - note.start)

        note_count = len(pitches)

        if note_count > 0:
            return {
                "note_count": note_count,
                "instrument_count": len(midi_obj.instruments),
                "is_drum": is_drum,
                "mean_pitch": float(np.mean(pitches)),
                "pitch_range": [int(min(pitches)), int(max(pitches))],
                "mean_velocity": float(np.mean(velocities)),
                "mean_duration": float(np.mean(durations)),
            }
        else:
            return {
                "note_count": 0,
                "instrument_count": len(midi_obj.instruments),
                "is_drum": is_drum,
                "mean_pitch": 0.0,
                "pitch_range": [0, 0],
                "mean_velocity": 0.0,
                "mean_duration": 0.0,
            }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代生成HuggingFace格式的样本"""
        for i, (audio_tensor, midi_obj) in enumerate(self.dataset):
            if self.max_samples is not None and i >= self.max_samples:
                break

            if i % 100 == 0:
                print(f'processing {i} samples')

            # 将音频张量转换为numpy数组
            audio_array = audio_tensor.numpy().astype(self.config.audio_dtype)

            # 创建样本，将音频存储为数组而不是Audio对象
            sample = {
                self.config.audio_column_name: audio_array,
                "audio_sampling_rate": self.config.sample_rate
            }

            if self.config.include_midi_bytes:
                sample[self.config.midi_column_name] = self._midi_to_bytes(midi_obj)

            if self.config.include_segment_info:
                sample[self.config.segment_info_column_name] = self._extract_segment_info(midi_obj)

            yield sample


def create_hf_features_from_config(config: HFDataConfig):
    """
    根据配置创建HuggingFace数据集的特征定义

    Args:
        config: 数据配置

    Returns:
        HuggingFace数据集特征定义
    """
    try:
        from datasets import Features, Array2D, Value, Sequence
    except ImportError:
        raise ImportError("请先安装datasets库：pip install datasets")

    # 使用Array2D存储音频数据，避免使用Audio特征
    features_dict = {
        config.audio_column_name: Sequence(Value(config.audio_dtype)),
        "audio_sampling_rate": Value("int32")
    }

    # MIDI字节数据
    if config.include_midi_bytes:
        features_dict[config.midi_column_name] = Value("binary")

    # 片段信息
    if config.include_segment_info:
        features_dict[config.segment_info_column_name] = {
            "note_count": Value("int64"),
            "instrument_count": Value("int64"),
            "is_drum": Value("bool"),
            "mean_pitch": Value("float32"),
            "pitch_range": Sequence(Value("int64"), length=2),
            "mean_velocity": Value("float32"),
            "mean_duration": Value("float32"),
        }

    return Features(features_dict)


def create_hf_dataset_from_midi_segment(
        midi_segment_dataset,
        config: HFDataConfig = None,
        max_samples: Optional[int] = None,
):
    """
    创建HuggingFace数据集

    Args:
        midi_segment_dataset: MIDISegmentDataset实例
        config: 数据配置
        max_samples: 最大样本数
        streaming: 是否创建流式数据集

    Returns:
        HuggingFace Dataset对象
    """
    try:
        from datasets import Dataset, IterableDataset
    except ImportError:
        raise ImportError("请先安装datasets库：pip install datasets")

    config = config or HFDataConfig()

    # 创建生成器
    generator = MIDISegmentToHFGenerator(
        midi_segment_dataset,
        config=config,
        max_samples=max_samples
    )

    # 创建特征定义
    features = create_hf_features_from_config(config)


    dataset = IterableDataset.from_generator(
        lambda: generator,
        features=features
    )


    return dataset


# 更新还原函数以处理新的格式
def restore_to_midi_dataset_format(
        hf_sample: Dict[str, Any],
        include_audio: bool = True,
        include_midi: bool = True,
) -> Dict[str, Any]:
    """
    将HuggingFace格式的样本还原为MIDI数据集格式

    Args:
        hf_sample: HuggingFace数据集样本
        include_audio: 是否包含音频张量
        include_midi: 是否包含MIDI对象

    Returns:
        包含原始格式数据的字典
    """
    result = {}

    if include_audio and "audio" in hf_sample:
        audio_array = hf_sample["audio"]

        # 确保是numpy数组
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # 获取采样率
        sample_rate = hf_sample.get("audio_sampling_rate", 24000)

        result["audio"] = torch.from_numpy(audio_array).float()
        result["sample_rate"] = sample_rate

    if include_midi and "midi_bytes" in hf_sample:
        midi_bytes = hf_sample["midi_bytes"]
        if isinstance(midi_bytes, bytes):
            try:
                midi_obj = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
                result["midi"] = midi_obj
            except Exception as e:
                logger.error(f"解析MIDI字节失败: {e}")
                result["midi"] = None

    if "segment_info" in hf_sample:
        result["segment_info"] = hf_sample["segment_info"]

    return result


def batch_restore_to_midi_dataset_format(
        hf_batch: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
        include_audio: bool = True,
        include_midi: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量将HuggingFace格式的数据还原为MIDI数据集格式

    Args:
        hf_batch: HuggingFace批次数据（列表或字典批次）
        include_audio: 是否包含音频张量
        include_midi: 是否包含MIDI对象

    Returns:
        原始格式数据的列表
    """
    if isinstance(hf_batch, dict):
        batch_size = len(next(iter(hf_batch.values())))
        hf_batch = [
            {key: values[i] for key, values in hf_batch.items()}
            for i in range(batch_size)
        ]

    return [
        restore_to_midi_dataset_format(
            sample,
            include_audio=include_audio,
            include_midi=include_midi,
        )
        for sample in hf_batch
    ]

class IterableMIDIDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset_iterator()

    def dataset_iterator(self):
        for sample in self.dataset:
            restored = restore_to_midi_dataset_format(sample)
            if restored.get("audio") is not None and restored.get("midi") is not None:
                yield restored["audio"], restored["midi"]


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

if __name__ == "__main__":
    import soundfile as sf
    # 示例1: 自定义Dataset\

    null_handler = logging.NullHandler()

    logger = logging.getLogger()

    logger.addHandler(null_handler)


    # 创建MIDI切片数据集
    midi_dataset = load_dataset("astune/mts_rl_dataset", streaming=True)

    # 创建torch数据集
    torch_dataset = IterableMIDIDataset(midi_dataset['train'])

    # 创建数据加载器 - 对于IterableDataset，shuffle必须为False
    dataloader = DataLoader(
        torch_dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False  # IterableDataset不支持shuffle
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