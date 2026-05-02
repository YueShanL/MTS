import random
from pathlib import Path
from typing import List, Dict, Optional, Generator

import librosa
import numpy as np
import torch
from guitarpro import Song, NoteEffect, Duration, Beat
from torch import Tensor
from torch.nn import functional

from model.gp_utils import duration_to_idx, idx_to_duration
from utils.GP5Generator import GuitarProGenerator, GuitarTechnique
from utils.mid2gp import MIDItoGP5Converter
from utils.mid_preprocessor import midi_to_audio_tensor


class AudioGuitarTabDataset(torch.utils.data.Dataset):
    """音频到吉他谱数据集"""
    def __init__(self,
                 audio_inputs: List[Tensor],
                 tab_data: List[Dict],  # 吉他谱数据
                 max_output_length: int = 128,
                 bpm: int = 120,
                 sample_rate: int = 24000,
                 sliding_window: int = 10,
                 segment: int = 16,
                 context_len = 8,
                 max_len:int = 200,
                 data_enhance = False):
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
        self.data_enhance = data_enhance

        # 验证数据
        assert len(audio_inputs) == len(tab_data), "音频和标签数量不匹配"

        self.init_cumulative_slices()

    def __len__(self):
        return (self.cumulative_slices[-1] * 3) if self.data_enhance else self.cumulative_slices[-1]

    def __getitem__(self, idx):
        if len(self.slice_counts) == 0 and len(self.audio_inputs) != 0:
            self.init_cumulative_slices()

        import bisect
        # 找到第一个 >= idx 的位置，然后减1得到 data_idx
        sub = 0
        if self.data_enhance:
            idx = idx // 3
            sub = idx % 3
        data_idx = bisect.bisect_right(self.cumulative_slices, idx) - 1
        slice_idx = idx - self.cumulative_slices[data_idx]

        audio_length = len(self.audio_inputs[data_idx])
        audio_start = slice_idx * self.step * self.sample_rate + self.context_len * self.sample_rate
        audio_end = min(audio_start + (self.segment - self.context_len) * self.sample_rate, audio_length)

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
                tab_data[key] = Tensor(padded[self.context_len * self.bpm // 60 * 4:]).to(torch.int64)
                context_data[key] = Tensor(padded[:self.context_len * self.bpm // 60 * 4]).to(torch.int64)

        # padding
        if len(input_audio) < (self.segment - self.context_len) * self.sample_rate:
            pad_length = (self.segment - self.context_len) * self.sample_rate - len(input_audio)
            # 使用torch.nn.functional.pad，更高效且与后续流程兼容
            input_audio = functional.pad(input_audio, (0, pad_length), mode='constant', value=0)

        result = {
            'audio_input': input_audio,
            'context_notes': context_data,
            'target_notes': tab_data
        }
        return self.enhance_data(sub, result)

    def enhance_data(self, idx, data):
        if idx == 1:
            semitones = random.choice([1, 2, 3, 4])
            data['context_notes'] = self.transpose_guitar_tab(data['context_notes'], semitones)
            data['target_notes'] = self.transpose_guitar_tab(data['target_notes'], semitones)
            data['audio_input'] = self.transpose_waveform(data['audio_input'], semitones, self.sample_rate)
        elif idx == 2:
            semitones = random.choice([-1, -2, -3, -4])
            data['context_notes'] = self.transpose_guitar_tab(data['context_notes'], semitones)
            data['target_notes'] = self.transpose_guitar_tab(data['target_notes'], semitones)
            data['audio_input'] = self.transpose_waveform(data['audio_input'], semitones, self.sample_rate)
        return data

    @staticmethod
    def transpose_waveform(waveform, semitones, sample_rate=24000):
        """
        使用librosa进行高质量的音频变调

        参数:
        - waveform: 音频波形，形状为 [batch, samples]
        - semitones: 变调半音数
        - sample_rate: 采样率

        返回:
        - 变调后的波形
        """
        batch_size, num_samples = waveform.shape
        transposed = torch.zeros_like(waveform)

        for b in range(batch_size):
            # 转换为numpy数组
            audio_np = waveform[b].cpu().numpy()

            # 使用librosa进行变调
            # 注意：librosa的pitch_shift返回的音频长度可能不同
            shifted = librosa.effects.pitch_shift(
                audio_np,
                sr=sample_rate,
                n_steps=semitones,
                bins_per_octave=12
            )

            # 确保长度匹配
            if len(shifted) > num_samples:
                shifted = shifted[:num_samples]
            elif len(shifted) < num_samples:
                # 填充
                padded = np.zeros(num_samples)
                padded[:len(shifted)] = shifted
                shifted = padded

            transposed[b] = torch.from_numpy(shifted).to(waveform.device)


        return transposed.to(waveform.device)

    @staticmethod
    def transpose_guitar_tab(data, semitones, max_fret=24):
        """
        向量化实现的直接法吉他变调（更高效）

        参数:
        - data: 输入数据
        - semitones: 变调半音数
        - max_fret: 最大品位

        返回:
        - 变调后的数据
        """
        # 复制数据
        result = {
            'duration': data['duration'].clone(),
            'fret': data['fret'].clone(),
            'technique': data['technique'].clone()
        }

        fret_tensor = result['fret']
        technique_tensor = result['technique']
        batch_size, target_length, num_strings = fret_tensor.shape

        # 1. 直接加减半音
        new_frets = fret_tensor + semitones

        # 2. 找出违规位置
        invalid_mask = (new_frets < 0) | (new_frets > max_fret)

        # 3. 创建合法的变调结果（先不处理违规位置）
        result['fret'] = new_frets

        # 4. 为每个违规位置尝试移动弦
        # 定义弦间音程差矩阵 [6, 6]，表示从弦i移动到弦j需要的半音调整
        # 正值表示需要增加的半音数，负值表示需要减少的半音数
        interval_matrix = torch.zeros((6, 6), device=fret_tensor.device)

        # 标准调弦：E(0) A(5) D(10) G(15) B(19) E(24)
        standard_tuning = torch.tensor([0, 5, 10, 15, 19, 24], device=fret_tensor.device)

        # 计算任意两根弦之间的音程差
        for i in range(6):
            for j in range(6):
                interval_matrix[i, j] = standard_tuning[j] - standard_tuning[i]

        # 5. 处理每个批次和时间步
        for b in range(batch_size):
            for t in range(target_length):
                # 获取当前位置的所有音符
                current_frets = new_frets[b, t].clone()
                current_tech = technique_tensor[b, t].clone()

                # 找出违规位置
                invalid_positions = invalid_mask[b, t]

                if invalid_positions.any():
                    # 对每个违规位置尝试修复
                    for s in torch.where(invalid_positions)[0]:
                        original_fret = fret_tensor[b, t, s]  # 原始品位

                        if original_fret == 0:  # 空弦，跳过
                            continue

                        # 尝试移动到其他弦
                        best_move = None
                        best_fret = None

                        # 检查所有其他弦
                        for other_s in range(6):
                            if other_s == s:
                                continue

                            # 计算在新弦上的品位
                            # 新品位 = 原始品位 + 变调 + 弦间音程差
                            interval = interval_matrix[s, other_s]
                            new_fret_on_other = original_fret + semitones + interval

                            # 检查是否合法
                            if 0 <= new_fret_on_other <= max_fret:
                                # 检查目标弦是否空闲或可替换
                                target_fret = current_frets[other_s]

                                if target_fret == 0 or (invalid_mask[b, t, other_s] and original_fret > 0):
                                    # 这是一个可行的移动
                                    if best_move is None or abs(other_s - s) < abs(best_move - s):
                                        # 选择最近的一根弦
                                        best_move = other_s
                                        best_fret = int(new_fret_on_other)

                        # 如果找到合适的移动
                        if best_move is not None:
                            # 执行移动
                            result['fret'][b, t, best_move] = best_fret
                            result['technique'][b, t, best_move] = technique_tensor[b, t, s]
                            result['fret'][b, t, s] = 0
                        else:
                            # 没有合适的移动，裁剪到合法范围
                            clamped = max(0, min(max_fret, original_fret + semitones))
                            result['fret'][b, t, s] = int(clamped)

        # 6. 最终确保所有值合法
        result['fret'] = torch.clamp(result['fret'], min=0, max=max_fret).long()

        return result

    def slide_data(self, audio, tab, type = "pt"):
        audio_length = len(audio)
        num_slices = max(1,
                         (audio_length - self.sliding_window * self.sample_rate) // (self.step * self.sample_rate))
        audio_inputs = []
        context_notes = []
        target_notes = []
        for slice_idx in range(num_slices):
            audio_start = slice_idx * self.step * self.sample_rate + self.context_len * self.sample_rate
            audio_end = min(audio_start + (self.segment - self.context_len) * self.sample_rate, audio_length)

            input_audio = audio[audio_start:audio_end]

            note_length = len(tab['duration'])
            notes_start = slice_idx * self.step * self.bpm // 60 * 4
            context_end = notes_start + self.context_len * self.bpm // 60 * 4
            notes_end = min(notes_start + self.segment * self.bpm // 60 * 4, note_length)

            tab_data = {}
            context_data = {}
            for key in tab.keys():
                tab_data[key] = tab[key][context_end:notes_end]
                context_data[key] = tab[key][notes_start:context_end]
                if len(tab_data[key]) + len(context_data[key]) < self.segment * self.bpm // 60 * 4:
                    pad_length = self.segment * self.bpm // 60 * 4 - len(tab_data[key])
                    pad_width = [(0, 0)] * tab_data[key].ndim
                    pad_width[0] = (0, pad_length)
                    padded = np.pad(tab_data[key], pad_width, mode='constant')
                    tab_data[key] = Tensor(padded[self.context_len * self.bpm // 60 * 4:]).to(torch.int64)
                    context_data[key] = Tensor(padded[:self.context_len * self.bpm // 60 * 4]).to(torch.int64)

                if type == 'py':
                    tab_data[key] = tab_data[key].tolist()
                    context_data[key] = context_data[key].tolist()

            # padding
            if len(input_audio) < (self.segment - self.context_len) * self.sample_rate:
                pad_length = (self.segment - self.context_len) * self.sample_rate - len(input_audio)
                # 使用torch.nn.functional.pad，更高效且与后续流程兼容
                input_audio = functional.pad(input_audio, (0, pad_length), mode='constant', value=0)

            if type == 'py':
                input_audio = input_audio.tolist()

            context_notes.append(context_data)
            target_notes.append(tab_data)
            audio_inputs.append(input_audio)

        return audio_inputs, context_notes, target_notes

    def stream_generator(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Generator[Dict, None, None]:
        """流式生成数据"""
        if end_idx is None:
            end_idx = len(self)

        for idx in range(start_idx, end_idx):
            example = self[idx]
            yield {
                'audio_input': example['audio_input'].tolist(),
                'context_notes': {k: a.tolist() for k, a in example['context_notes'].items()},
                'target_notes': {k: a.tolist() for k, a in example['target_notes'].items()}
            }



    def to(self, device):
        for a in self.audio_inputs:
            a.to(device)

        for b in self.tab_data:
            for c in b.values():
                c.to(device)

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
    def generator_from_path(cls, mid_path:str, start = 0, limit = None, type = 'pt'):
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

        limit = limit if limit is not None else (len(midi_files) - start)

        for idx, f in enumerate(midi_files[start:start + limit]):
            if idx >= limit:
                break
            song = convertor.convert_midi_to_gp5(midi_path=f.__str__(), post_process=False)
            encoded = cls.encode_tab_sequence(dataset, song=song)
            duration = len(encoded['duration']) / 8
            wav, _ = midi_to_audio_tensor(f.__str__(), sr=dataset.sample_rate, duration=duration, debug=False)
            a, c, t = dataset.slide_data(wav, encoded, type=type)
            for i in  range(len(a)):
                yield {
                    'audio_input': a[i],
                    'context_notes': c[i],
                    'target_notes': t[i]
                }


    @classmethod
    def create_from_path(cls, mid_path:str, start = 0, limit = 1000):
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
        soundfont_path = "../asset/GeneralUser-GS.sf2"
        for idx, f in enumerate(midi_files[start:start + limit]):
            if idx >= limit:
                break
            song = convertor.convert_midi_to_gp5(midi_path=f.__str__(), post_process=False)
            encoded = cls.encode_tab_sequence(dataset, song=song)
            tab_data.append(encoded)
            duration = len(encoded['duration']) / 8
            wav, _ = midi_to_audio_tensor(f.__str__(), sr=dataset.sample_rate, duration=duration, debug=False)
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
                    v = 127
                    if tab_sequence.get('velocity_tokens') is not None:
                        v = tab_sequence['velocity_tokens'][idx][string]
                    f = fret.tolist() if isinstance(fret, Tensor) else fret
                    t = tab_sequence['technique'][idx][string]
                    t = t.tolist() if isinstance(t, Tensor) else t
                    v = v.tolist() if isinstance(v, Tensor) else v
                    generator.add_note(song, string + 1, f, dur, technique=GuitarTechnique(t), position=position, measure_index=measure_idx, velocity=v)
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
    dataset = AudioGuitarTabDataset(audio_inputs, tab_data)
    decoded = decode(tab_sequence, tempo = 120, post_process=False)
    result = dataset.encode_tab_sequence(decoded)
    duration_match = torch.all(duration == result['duration'])
    fret_match = torch.all(fret == result['fret'])
    technique_match = torch.all(technique == result['technique'])
    assert duration_match and fret_match and technique_match

if __name__ == "__main__":
    main()