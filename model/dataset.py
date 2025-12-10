from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
import torch
from guitarpro import Song, NoteEffect, Duration, Beat
from torch import Tensor
from transformers import Wav2Vec2Processor

from model.gp_utils import duration_to_idx, idx_to_duration
from utils.GP5Generator import GuitarProGenerator, GuitarTechnique
from utils.mid2gp import MIDItoGP5Converter
from utils.mid_preprocessor import midi_to_audio_tensor


class AudioProcessor:
    """音频预处理工具类"""

    def __init__(self, sample_rate=16000, max_duration=30):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = sample_rate * max_duration

        # 加载Wav2Vec2处理器
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def load_and_preprocess(self, audio_path):
        """加载并预处理音频文件"""
        # 使用librosa加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 标准化长度
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            # 填充静音
            padding = np.zeros(self.max_samples - len(audio))
            audio = np.concatenate([audio, padding])

        # 转换为张量
        audio_tensor = torch.from_numpy(audio).float()

        # 使用Wav2Vec2处理器
        inputs = self.processor(
            audio_tensor,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        return inputs.input_values.squeeze()

    def extract_features(self, audio_tensor):
        """提取音频特征"""
        # 这里可以添加额外的音频特征提取
        features = {
            'waveform': audio_tensor,
            'sr': self.sample_rate
        }
        return features

class AudioGuitarTabDataset(torch.utils.data.Dataset):
    """音频到吉他谱数据集"""
    def __init__(self,
                 audio_inputs: List[Tensor],
                 tab_data: List[Dict],  # 吉他谱数据
                 max_output_length: int = 128,
                 bpm: int = 120,
                 sample_rate: int = 16000,
                 sliding_window: int = 10,
                 segment: int = 16,
                 context_len = 8,
                 max_len:int = 200):
        self.context_len = context_len
        self.cumulative_slices = None
        self.slice_counts = None
        self.segment = segment
        self.step = segment - sliding_window
        self.sliding_window = sliding_window
        self.audio_inputs = audio_inputs
        self.tab_data = tab_data
        self.sample_rate = sample_rate
        self.max_output_length = max_output_length
        self.bpm = bpm
        self.max_len = max_len
        self.dataset_len = max_len

        # 验证数据
        assert len(audio_inputs) == len(tab_data), "音频和标签数量不匹配"

        self.init_cumulative_slices()

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, idx):
        if len(self.slice_counts) == 0 and len(self.audio_inputs) != 0:
            self.init_cumulative_slices()

        # Get raw data
        slice_idx = 0
        data_idx = 0
        for i in range(len(self.cumulative_slices) - 1):
            if self.cumulative_slices[i] <= idx < self.cumulative_slices[i + 1]:
                data_idx = i
                slice_idx = idx - self.cumulative_slices[i]
                break

        audio_length = len(self.audio_inputs[data_idx])
        audio_start = min(slice_idx * (self.step + self.context_len) * self.sample_rate, audio_length)
        audio_end = min(audio_start + self.segment * self.sample_rate, audio_length)

        input_audio = self.audio_inputs[data_idx][audio_start:audio_end]

        note_length = len(self.tab_data[data_idx]['duration'])
        notes_start = slice_idx * self.step * self.bpm // 60 * 4
        context_end = notes_start + self.context_len * self.bpm // 60 * 4
        notes_end = min(notes_start + self.segment * self.bpm // 60 * 4, note_length)

        tab_data = {}
        context_data = {}
        for key in self.tab_data[data_idx].keys():
            tab_data[key] = self.tab_data[data_idx][key][context_end:notes_end]
            context_data[key] = self.tab_data[data_idx][key][notes_start:context_end]
            if len(tab_data[key]) + len(context_data[key]) < self.segment * self.bpm // 60 * 4:
                pad_length = self.segment * self.bpm // 60 * 4 - len(tab_data[key])
                pad_width = [(0, 0)] * tab_data[key].ndim
                pad_width[0] = (0, pad_length)
                padded = np.pad(tab_data[key], pad_width, mode='constant')
                tab_data[key] = Tensor(padded[self.context_len * self.bpm // 60 * 4:])
                context_data[key] = Tensor(padded[:self.context_len * self.bpm // 60 * 4])

        # padding
        if len(input_audio) < self.segment * self.sample_rate:
            pad_length = self.segment * self.sample_rate - len(input_audio)
            input_audio = Tensor(np.pad(input_audio, (0, pad_length), mode='constant'))

        return {
            'audio_input': input_audio,
            'context_notes': context_data,
            'target_notes': tab_data
        }

    def init_cumulative_slices(self):
        self.dataset_len = min(self.max_len, len(self.audio_inputs))
        self.slice_counts = []
        for i in range(self.dataset_len):
            audio_length = len(self.audio_inputs[i])
            num_slices = max(1, (audio_length - self.sliding_window * self.sample_rate) // (self.step * self.sample_rate))
            self.slice_counts.append(num_slices)

        self.cumulative_slices = [0]
        for count in self.slice_counts:
            self.cumulative_slices.append(self.cumulative_slices[-1] + count)
        return

    @classmethod
    def create_from_path(cls, mid_path:str, limit = 1000):
        extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []

        for ext in extensions:
            midi_files.extend(Path(mid_path).rglob(f'*{ext}'))

        midi_files = sorted(list(set(midi_files)))

        print(f"find {len(midi_files)} .MIDI files")

        audio_inputs = []
        tab_data = []
        dataset = AudioGuitarTabDataset(audio_inputs, tab_data)

        convertor = MIDItoGP5Converter(GuitarProGenerator())

        for idx, f in enumerate(midi_files):
            if idx >= limit:
                break
            song = convertor.convert_midi_to_gp5(midi_path=f.__str__(), post_process=False)
            tab_data.append(cls.encode_tab_sequence(dataset, song=song))
            wav, _ = midi_to_audio_tensor(f.__str__(), dataset.sample_rate, None, False)
            audio_inputs.append(wav)

        dataset.init_cumulative_slices()

        print(f'generate dataset with length {len(dataset)} at samplerate {dataset.sample_rate}')

        return dataset, {
            'audio_input': audio_inputs,
            'target_notes': tab_data
        }

    def encode_tab_sequence(self, song:Song):
        """编码吉他谱序列为模型标签"""

        size = len(song.tracks[0].measures) * 16
        tab_sequence:List[Beat|None] = [None] * size
        for m in song.tracks[0].measures:
            for b in m.voices[0].beats:
                tab_sequence[int((b.start - 4800) / Duration.quarterTime * 4)] = b

        for b in tab_sequence:
            if b is not None:
                assert (b.startInMeasure % (Duration.quarterTime / 4)) == 0, f'expecting all beats quantized to 16 but get {b.startInMeasure}'

        # 初始化标签
        max_len = size

        labels = {
            'fret': torch.zeros([max_len, 6], dtype=torch.long),
            'duration': torch.zeros(max_len, dtype=torch.long),
            'technique': torch.zeros([max_len, 6], dtype=torch.long),
        }

        # 填充标签
        for i, beat in enumerate(tab_sequence[:max_len]):
            position = torch.full([6], 25)
            tech = torch.full([6], 0)
            if beat is not None:
                for note in beat.notes:
                    position[note.string - 1] = note.value
                    tech[note.string - 1] = self._effect_map(note.effect)
                labels['duration'][i] = duration_to_idx(beat.duration)
            else:
                labels['duration'][i] = 0
            labels['fret'][i] = position
            labels['technique'][i] = tech

        return labels

    def _effect_map(self, effect: NoteEffect) -> int:
        """
        将NoteEffect映射到技巧索引

        Args:
            effect: NoteEffect对象

        Returns:
            技巧索引 (0-13)
        """

        # 按优先级检查效果（从高到低）

        # 1. 检查闷音 - 最高优先级
        if effect.ghostNote:
            return GuitarTechnique.MUTE.value

        # 2. 检查手掌闷音
        if effect.palmMute:
            return GuitarTechnique['PALM_MUTE'].value

        # 3. 检查击弦
        if effect.hammer:
            return GuitarTechnique['HAMMER_ON'].value

        # 4. 检查泛音
        if effect.harmonic is not None:
            # 根据泛音类型返回相应的技巧
            if effect.harmonic.type == 2:  # 点弦泛音
                return GuitarTechnique.ARTIFICIAL_HARMONIC.value
            elif effect.harmonic.type == 3:
                return GuitarTechnique.TAPPED_HARMONIC.value
            elif effect.harmonic.type == 4:  # 掐泛音
                return GuitarTechnique['PINCH_HARMONIC'].value
            elif effect.harmonic.type == 5:  # 半泛音
                return GuitarTechnique['SEMI_HARMONIC'].value
            else:  # 自然泛音或人工泛音
                return GuitarTechnique.NATURAL_HARMONIC.value

        # 5. 检查滑音
        if effect.slides and len(effect.slides) > 0:
            return GuitarTechnique['SLIDE'].value

        # 6. 检查弯音
        if effect.bend is not None:
            return GuitarTechnique['BEND'].value

        # 7. 检查颤音
        if effect.vibrato or effect.trill is not None:
            return GuitarTechnique['VIBRATO'].value

        # 8. 检查震音
        if effect.tremoloPicking is not None:
            return GuitarTechnique['TREMOLO'].value

        # 9. 检查断奏
        if effect.staccato:
            return GuitarTechnique['NORMAL'].value  # 断奏暂时映射到NORMAL

        # 10. 检查装饰音
        if effect.grace is not None:
            return GuitarTechnique['NORMAL'].value  # 装饰音暂时映射到NORMAL

        # 11. 检查重音标记
        if effect.accentuatedNote or effect.heavyAccentuatedNote:
            return GuitarTechnique['NORMAL'].value  # 重音暂时映射到NORMAL

        # 12. 默认返回普通
        return GuitarTechnique['NORMAL'].value

def decode(tab_sequence:Dict, tempo: int = 120, post_process=True):
    generator = GuitarProGenerator()
    song = generator.create_empty_song(tempo = tempo)
    size = len(tab_sequence['duration'])
    for idx in range(size):
        if tab_sequence['duration'][idx] > 0:
            dur = idx_to_duration(tab_sequence['duration'][idx] - 1)
            for string, fret in enumerate(tab_sequence['fret'][idx]):
                measure_idx = idx // 16
                position = (idx % 16) / 4
                if fret != 25:
                    generator.add_note(song, string + 1, fret.tolist(), dur, technique=GuitarTechnique(tab_sequence['technique'][idx][string].tolist()), position=position, measure_index=measure_idx)
    if post_process:
        generator.post_process(song)
    return song

def main():
    #torch.manual_seed(42)
    #np.random.seed(42)

    seq_len = 32  # 2 * 16

    duration = torch.randint(0, 13, [seq_len])  # 0-12
    fret = torch.randint(0, 26, (seq_len, 6))  # 0-25
    technique = torch.randint(0, 14, (seq_len, 6))  # 0-13

    #setup for ignored value (on duration = 0 or fret = 25)
    fret[duration == 0] = Tensor([25] * 6).to(torch.long)
    technique[duration == 0] = Tensor([0] * 6).to(torch.long)
    technique[fret == 25] = 0

    #replace invalid mapping
    technique[technique == 2] = 1

    # tab_sequence
    tab_sequence = {
        'duration': duration.tolist(),
        'fret': fret,
        'technique': technique
    }

    audio_inputs = []
    tab_data = []
    dataset = AudioGuitarTabDataset(audio_inputs, tab_data, AudioProcessor())
    decoded = decode(tab_sequence, tempo = 120, post_process=False)
    result = dataset.encode_tab_sequence(decoded)
    duration_match = torch.all(duration == result['duration'])
    fret_match = torch.all(fret == result['fret'])
    technique_match = torch.all(technique == result['technique'])
    assert duration_match and fret_match and technique_match

if __name__ == "__main__":
    main()