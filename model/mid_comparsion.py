"""
MIDI相似度比较工具包
集成多种比较方法，从精确匹配到风格相似度分析
"""

import os
import warnings

import numpy as np
import pretty_midi
from pretty_midi import PrettyMIDI
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

from dtaidistance import dtw


class MIDISimilarityToolkit:
    """
    MIDI相似度比较工具包
    集成多种比较方法，支持从精确匹配到风格相似度分析
    """

    def __init__(self, method='balanced', weights=None):
        """
        初始化比较器

        Args:
            method: 比较方法，可选:
                - 'exact': 精确匹配（音符级）
                - 'balanced': 平衡方法（默认）
                - 'style': 风格相似度
                - 'fast': 快速比较（性能优先）
                - 'melodic': 旋律轮廓比较
            weights: 自定义权重，如果为None则使用预设权重
        """
        self.method = method
        self.cache = {}  # 特征缓存

        # 预设权重配置
        self.preset_weights = {
            'exact': {
                'pitch_distribution': 0.3,
                'rhythm_distribution': 0.3,
                'note_alignment': 0.2,
                'velocity_distribution': 0.1,
                'timing_precision': 0.1
            },
            'balanced': {
                'pitch_distribution': 0.25,
                'rhythm_distribution': 0.25,
                'melodic_contour': 0.2,
                'harmonic_content': 0.2,
                'structure': 0.1
            },
            'style': {
                'melodic_shape': 0.30,
                'rhythmic_groove': 0.25,
                'harmonic_flavor': 0.25,
                'structural_pattern': 0.15,
                'expressive_intensity': 0.05
            },
            'fast': {
                'pitch_hist': 0.4,
                'duration_mean': 0.2,
                'note_count': 0.2,
                'time_span': 0.2
            },
            'melodic': {
                'contour_similarity': 0.4,
                'pitch_range': 0.2,
                'interval_distribution': 0.2,
                'rhythmic_pattern': 0.2
            }
        }

        # 使用预设权重或自定义权重
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.preset_weights.get(method, self.preset_weights['balanced'])

    def compare(self, midi_path1, midi_path2, use_cache=True):
        """
        比较两个MIDI文件的相似度

        Args:
            midi_path1: 第一个MIDI文件路径
            midi_path2: 第二个MIDI文件路径
            use_cache: 是否使用特征缓存

        Returns:
            float: 相似度分数（0-1），越高越相似
        """
        # 生成缓存键
        cache_key = f"{midi_path1}_{midi_path2}_{self.method}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # 加载MIDI文件
        try:
            if isinstance(midi_path1, str):
                midi1 = pretty_midi.PrettyMIDI(midi_path1)
            elif isinstance(midi_path1, PrettyMIDI):
                midi1 = midi_path1
            if isinstance(midi_path2, str):
                midi2 = pretty_midi.PrettyMIDI(midi_path2)
            elif isinstance(midi_path2, PrettyMIDI):
                midi2 = midi_path2
        except Exception as e:
            print(f"加载MIDI文件失败: {e}")
            return 0.0

        # 根据方法选择比较策略
        if self.method == 'exact':
            similarity = self._compare_exact(midi1, midi2)
        elif self.method == 'style':
            similarity = self._compare_style(midi1, midi2)
        elif self.method == 'fast':
            similarity = self._compare_fast(midi1, midi2)
        elif self.method == 'melodic':
            similarity = self._compare_melodic(midi1, midi2)
        else:  # balanced
            similarity = self._compare_balanced(midi1, midi2)

        # 缓存结果
        if use_cache:
            self.cache[cache_key] = similarity

        return similarity

    def batch_compare(self, midi_paths1, midi_paths2, use_cache=True):
        """
        批量比较MIDI文件

        Args:
            midi_paths1: 第一个MIDI文件路径列表
            midi_paths2: 第二个MIDI文件路径列表
            use_cache: 是否使用特征缓存

        Returns:
            list: 相似度分数列表
        """
        assert len(midi_paths1) == len(midi_paths2), "两个列表长度必须相同"

        similarities = []
        for path1, path2 in zip(midi_paths1, midi_paths2):
            similarity = self.compare(path1, path2, use_cache)
            similarities.append(similarity)

        return similarities

    def _compare_exact(self, midi1, midi2):
        """精确匹配比较"""
        # 提取所有音符
        notes1 = self._extract_all_notes(midi1)
        notes2 = self._extract_all_notes(midi2)

        if len(notes1) == 0 or len(notes2) == 0:
            return 0.0

        similarities = {}

        # 1. 音高分布相似度
        similarities['pitch_distribution'] = self._compare_pitch_distribution(notes1, notes2)

        # 2. 节奏分布相似度
        similarities['rhythm_distribution'] = self._compare_rhythm_distribution(notes1, notes2)

        # 3. 音符对齐相似度（时间位置）
        similarities['note_alignment'] = self._compare_note_alignment(notes1, notes2)

        # 4. 力度分布相似度
        similarities['velocity_distribution'] = self._compare_velocity_distribution(notes1, notes2)

        # 5. 时间精度相似度
        similarities['timing_precision'] = self._compare_timing_precision(notes1, notes2)

        # 加权组合
        total_similarity = sum(
            self.weights[k] * similarities[k]
            for k in self.weights
        )

        return total_similarity

    def _compare_balanced(self, midi1, midi2):
        """平衡比较方法"""
        notes1 = self._extract_all_notes(midi1)
        notes2 = self._extract_all_notes(midi2)

        if len(notes1) == 0 or len(notes2) == 0:
            return 0.0

        similarities = {}

        # 1. 音高分布相似度
        similarities['pitch_distribution'] = self._compare_pitch_distribution(notes1, notes2)

        # 2. 节奏分布相似度
        similarities['rhythm_distribution'] = self._compare_rhythm_distribution(notes1, notes2)

        # 3. 旋律轮廓相似度
        similarities['melodic_contour'] = self._compare_melodic_contour(notes1, notes2)

        # 4. 和声内容相似度
        similarities['harmonic_content'] = self._compare_harmonic_content(notes1, notes2)

        # 5. 结构相似度
        similarities['structure'] = self._compare_structure(notes1, notes2)

        # 加权组合
        total_similarity = sum(
            self.weights[k] * similarities[k]
            for k in self.weights
        )

        return total_similarity

    def _compare_style(self, midi1, midi2):
        """风格相似度比较"""
        notes1 = self._extract_all_notes(midi1)
        notes2 = self._extract_all_notes(midi2)

        if len(notes1) < 10 or len(notes2) < 10:
            # 音符太少，直接返回中等相似度，避免回退导致键名错误
            return 0.5

        similarities = {}

        # 提取风格特征
        melodic_feat1 = self._extract_melodic_shape_features(notes1)
        melodic_feat2 = self._extract_melodic_shape_features(notes2)
        similarities['melodic_shape'] = self._compare_melodic_shape(melodic_feat1, melodic_feat2)

        rhythm_feat1 = self._extract_rhythmic_groove_features(notes1)
        rhythm_feat2 = self._extract_rhythmic_groove_features(notes2)
        similarities['rhythmic_groove'] = self._compare_rhythmic_groove(rhythm_feat1, rhythm_feat2)

        harmony_feat1 = self._extract_harmonic_flavor_features(notes1)
        harmony_feat2 = self._extract_harmonic_flavor_features(notes2)
        similarities['harmonic_flavor'] = self._compare_harmonic_flavor(harmony_feat1, harmony_feat2)

        structure_feat1 = self._extract_structural_pattern_features(notes1)
        structure_feat2 = self._extract_structural_pattern_features(notes2)
        similarities['structural_pattern'] = self._compare_structural_pattern(structure_feat1, structure_feat2)

        expressive_feat1 = self._extract_expressive_intensity_features(notes1)
        expressive_feat2 = self._extract_expressive_intensity_features(notes2)
        similarities['expressive_intensity'] = self._compare_expressive_intensity(expressive_feat1, expressive_feat2)

        # 加权组合
        total_similarity = sum(
            self.weights.get(k, 0) * similarities.get(k, 0)
            for k in self.weights
        )

        return total_similarity

    def _compare_fast(self, midi1, midi2):
        """快速比较（性能优先）"""
        # 提取简化特征
        feat1 = self._extract_quick_features(midi1)
        feat2 = self._extract_quick_features(midi2)

        # 快速比较
        return self._quick_compare(feat1, feat2)

    def _compare_melodic(self, midi1, midi2):

        # 使用DTW比较旋律轮廓
        contour1 = self._extract_melody_contour(midi1)
        contour2 = self._extract_melody_contour(midi2)

        if len(contour1) < 2 or len(contour2) < 2:
            return 0.5

        # 计算DTW距离
        if contour1.ndim == 1:
            contour1 = contour1.reshape(-1, 1)
        if contour2.ndim == 1:
            contour2 = contour2.reshape(-1, 1)

        distance = dtw.distance(contour1, contour2)

        # 归一化距离到相似度
        max_range1 = max(contour1) - min(contour1) if len(contour1) > 0 else 1
        max_range2 = max(contour2) - min(contour2) if len(contour2) > 0 else 1
        max_possible_distance = (max_range1 + max_range2) * min(len(contour1), len(contour2))

        if max_possible_distance == 0:
            similarity = 1.0
        else:
            similarity = 1.0 - (distance / max_possible_distance)

        return max(0.0, min(1.0, similarity))

    # ================== 特征提取方法 ==================

    def _extract_all_notes(self, midi_obj):
        """提取MIDI中所有音符"""
        notes = []
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'duration': note.end - note.start,
                    'velocity': note.velocity
                })
        return notes

    def _extract_quick_features(self, midi_obj):
        """快速提取关键特征"""
        notes = []
        for inst in midi_obj.instruments[:1]:  # 只处理第一个音轨
            for note in inst.notes[:500]:  # 限制音符数量
                notes.append({
                    'pitch': note.pitch,
                    'duration': note.end - note.start
                })

        if notes:
            # 音高直方图
            pitch_hist = np.zeros(12)
            for note in notes:
                pitch_hist[note['pitch'] % 12] += 1
            pitch_hist = pitch_hist / pitch_hist.sum() if pitch_hist.sum() > 0 else pitch_hist

            # 时长特征
            durations = [note['duration'] for note in notes]
            duration_mean = np.mean(durations) if durations else 0

            features = {
                'pitch_hist': pitch_hist,
                'duration_mean': duration_mean,
                'note_count': len(notes),
                'time_span': midi_obj.get_end_time()
            }
        else:
            features = {
                'pitch_hist': np.zeros(12),
                'duration_mean': 0,
                'note_count': 0,
                'time_span': 0
            }

        return features

    def _quick_compare(self, feat1, feat2):
        """快速比较特征"""
        if feat1['note_count'] == 0 or feat2['note_count'] == 0:
            return 0.0

        # 1. 音高分布相似度
        pitch_sim = 1.0 - cosine(feat1['pitch_hist'], feat2['pitch_hist'])

        # 2. 时长特征相似度
        if feat1['duration_mean'] > 0 and feat2['duration_mean'] > 0:
            duration_ratio = min(feat1['duration_mean'], feat2['duration_mean']) / \
                             max(feat1['duration_mean'], feat2['duration_mean'])
        else:
            duration_ratio = 0.5

        # 3. 音符数量相似度
        count_ratio = min(feat1['note_count'], feat2['note_count']) / \
                      max(feat1['note_count'], feat2['note_count'])

        # 4. 时间跨度相似度
        if feat1['time_span'] > 0 and feat2['time_span'] > 0:
            time_ratio = min(feat1['time_span'], feat2['time_span']) / \
                         max(feat1['time_span'], feat2['time_span'])
        else:
            time_ratio = 0.5

        # 加权组合
        weights = [0.4, 0.2, 0.2, 0.2]
        similarities = [pitch_sim, duration_ratio, count_ratio, time_ratio]

        total_sim = sum(w * s for w, s in zip(weights, similarities))

        return total_sim

    # ================== 精确比较组件 ==================

    def _compare_pitch_distribution(self, notes1, notes2):
        """比较音高分布"""
        # 计算音高直方图（12音级）
        pitch_hist1 = np.zeros(12)
        pitch_hist2 = np.zeros(12)

        for note in notes1:
            pitch_class = note['pitch'] % 12
            pitch_hist1[pitch_class] += note['duration']

        for note in notes2:
            pitch_class = note['pitch'] % 12
            pitch_hist2[pitch_class] += note['duration']

        # 归一化
        if pitch_hist1.sum() > 0:
            pitch_hist1 = pitch_hist1 / pitch_hist1.sum()
        if pitch_hist2.sum() > 0:
            pitch_hist2 = pitch_hist2 / pitch_hist2.sum()

        if pitch_hist1.sum() == 0 or pitch_hist2.sum() == 0:
            return 0.0

        return 1.0 - cosine(pitch_hist1, pitch_hist2)

    def _compare_rhythm_distribution(self, notes1, notes2):
        """比较节奏分布"""
        durations1 = [note['duration'] for note in notes1]
        durations2 = [note['duration'] for note in notes2]

        if len(durations1) == 0 or len(durations2) == 0:
            return 0.0

        # 创建直方图
        all_durations = durations1 + durations2
        min_dur = min(all_durations)
        max_dur = max(all_durations)

        bins = 20
        hist1, _ = np.histogram(durations1, bins=bins, range=(min_dur, max_dur), density=True)
        hist2, _ = np.histogram(durations2, bins=bins, range=(min_dur, max_dur), density=True)

        # 巴氏系数
        bc = np.sum(np.sqrt(hist1 * hist2))
        return bc

    def _compare_note_alignment(self, notes1, notes2):
        """比较音符对齐"""
        # 按开始时间排序
        notes1_sorted = sorted(notes1, key=lambda x: x['start'])
        notes2_sorted = sorted(notes2, key=lambda x: x['start'])

        if len(notes1_sorted) == 0 or len(notes2_sorted) == 0:
            return 0.0

        # 采样时间点，检查是否有音符
        time_points = np.linspace(
            min(notes1_sorted[0]['start'], notes2_sorted[0]['start']),
            max(notes1_sorted[-1]['end'], notes2_sorted[-1]['end']),
            100
        )

        matches = 0
        for t in time_points:
            # 检查第一个MIDI在t时刻是否有音符
            has_note1 = any(n['start'] <= t <= n['end'] for n in notes1_sorted)
            has_note2 = any(n['start'] <= t <= n['end'] for n in notes2_sorted)

            if has_note1 == has_note2:
                matches += 1

        return matches / len(time_points)

    def _compare_velocity_distribution(self, notes1, notes2):
        """比较力度分布"""
        velocities1 = [note['velocity'] for note in notes1]
        velocities2 = [note['velocity'] for note in notes2]

        if len(velocities1) == 0 or len(velocities2) == 0:
            return 0.5

        # 计算直方图
        hist1, bins = np.histogram(velocities1, bins=10, range=(0, 128), density=True)
        hist2, _ = np.histogram(velocities2, bins=bins, density=True)

        # 余弦相似度
        return 1.0 - cosine(hist1, hist2)

    def _compare_timing_precision(self, notes1, notes2):
        """比较时间精度"""
        # 提取音符开始时间
        starts1 = sorted([note['start'] for note in notes1])
        starts2 = sorted([note['start'] for note in notes2])

        if len(starts1) < 2 or len(starts2) < 2:
            return 0.5

        # 计算时间间隔
        intervals1 = np.diff(starts1)
        intervals2 = np.diff(starts2)

        # 比较间隔分布
        hist1, bins = np.histogram(intervals1, bins=10, density=True)
        hist2, _ = np.histogram(intervals2, bins=bins, density=True)

        return 1.0 - cosine(hist1, hist2)

    # ================== 平衡比较组件 ==================

    def _compare_melodic_contour(self, notes1, notes2):
        """比较旋律轮廓"""
        # 提取旋律线
        melody1 = self._extract_melody_line(notes1)
        melody2 = self._extract_melody_line(notes2)

        if len(melody1) < 2 or len(melody2) < 2:
            return 0.5

        # 计算轮廓
        contour1 = np.sign(np.diff([note['pitch'] for note in melody1]))
        contour2 = np.sign(np.diff([note['pitch'] for note in melody2]))

        # 取较短的长度
        min_len = min(len(contour1), len(contour2))
        if min_len == 0:
            return 0.0

        # 计算匹配比例
        matches = np.sum(contour1[:min_len] == contour2[:min_len])
        return matches / min_len

    def _extract_melody_line(self, notes):
        """提取旋律线"""
        if not notes:
            return []

        # 简单方法：每0.5秒取一个最高音
        sorted_notes = sorted(notes, key=lambda x: x['start'])
        melody = []
        time_window = 0.5
        current_time = sorted_notes[0]['start']
        end_time = sorted_notes[-1]['end']

        while current_time < end_time:
            window_notes = [
                note for note in sorted_notes
                if current_time <= note['start'] < current_time + time_window
            ]

            if window_notes:
                highest_note = max(window_notes, key=lambda x: x['pitch'])
                melody.append(highest_note)

            current_time += time_window

        return melody

    def _compare_harmonic_content(self, notes1, notes2):
        """比较和声内容"""
        # 提取和弦集合
        chord_sets1 = self._extract_chord_sets(notes1)
        chord_sets2 = self._extract_chord_sets(notes2)

        if not chord_sets1 or not chord_sets2:
            return 0.5

        # Jaccard相似度
        set1 = set(chord_sets1)
        set2 = set(chord_sets2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def _extract_chord_sets(self, notes):
        """提取和弦集合"""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        chord_sets = []
        time_quantize = 0.25

        current_time = sorted_notes[0]['start']
        end_time = sorted_notes[-1]['end']

        while current_time < end_time:
            chord_notes = [
                note for note in sorted_notes
                if current_time <= note['start'] < current_time + time_quantize
            ]

            if chord_notes:
                pitch_classes = tuple(sorted(set(note['pitch'] % 12 for note in chord_notes)))
                if len(pitch_classes) >= 2:  # 至少2个音才视为和弦
                    chord_sets.append(pitch_classes)

            current_time += time_quantize

        return chord_sets

    def _compare_structure(self, notes1, notes2):
        """比较结构"""
        # 计算音符密度
        density1 = self._compute_note_density(notes1)
        density2 = self._compute_note_density(notes2)

        if len(density1) < 2 or len(density2) < 2:
            return 0.5

        # 取相同长度
        min_len = min(len(density1), len(density2))
        density1 = density1[:min_len]
        density2 = density2[:min_len]

        # 计算相关系数
        if np.std(density1) == 0 or np.std(density2) == 0:
            return 0.5

        corr, _ = pearsonr(density1, density2)
        return (corr + 1) / 2

    def _compute_note_density(self, notes, window_size=1.0):
        """计算音符密度时间序列"""
        if not notes:
            return np.array([0])

        start_time = min(note['start'] for note in notes)
        end_time = max(note['end'] for note in notes)

        num_windows = int(np.ceil((end_time - start_time) / window_size))
        density = np.zeros(num_windows)

        for note in notes:
            start_window = int((note['start'] - start_time) / window_size)
            end_window = int((note['end'] - start_time) / window_size)

            start_window = max(0, min(start_window, num_windows - 1))
            end_window = max(0, min(end_window, num_windows - 1))

            for w in range(start_window, end_window + 1):
                density[w] += 1

        return density

    # ================== 风格比较组件 ==================

    def _extract_melodic_shape_features(self, notes):
        """提取旋律形状特征"""
        if len(notes) < 3:
            return {'contour': [], 'range': 0, 'step_ratio': 0, 'leap_ratio': 0}

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        pitches = [note['pitch'] for note in sorted_notes[:50]]

        if len(pitches) < 2:
            return {'contour': [], 'range': 0, 'step_ratio': 0, 'leap_ratio': 0}

        # 旋律轮廓
        contour = []
        for i in range(1, len(pitches)):
            if pitches[i] > pitches[i - 1]:
                contour.append(1)
            elif pitches[i] < pitches[i - 1]:
                contour.append(-1)
            else:
                contour.append(0)

        # 旋律范围
        pitch_range = max(pitches) - min(pitches) if pitches else 0

        # 音程类型
        intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
        step_motion = sum(1 for i in intervals if i <= 2)
        leap_motion = sum(1 for i in intervals if i > 2)

        step_ratio = step_motion / len(intervals) if intervals else 0
        leap_ratio = leap_motion / len(intervals) if intervals else 0

        return {
            'contour': contour,
            'range': pitch_range,
            'step_ratio': step_ratio,
            'leap_ratio': leap_ratio
        }

    def _compare_melodic_shape(self, feat1, feat2):
        """比较旋律形状"""
        # 轮廓序列相似度
        contour_sim = self._compare_contour_sequences(feat1['contour'], feat2['contour'])

        # 音域相似度
        if feat1['range'] > 0 and feat2['range'] > 0:
            range_sim = 1.0 - min(1.0, abs(feat1['range'] - feat2['range']) / 24)
        else:
            range_sim = 0.5

        # 运动类型相似度
        motion_sim = (1.0 - abs(feat1['step_ratio'] - feat2['step_ratio'])) * 0.5 + \
                     (1.0 - abs(feat1['leap_ratio'] - feat2['leap_ratio'])) * 0.5

        return contour_sim * 0.5 + range_sim * 0.3 + motion_sim * 0.2

    def _compare_contour_sequences(self, contour1, contour2):
        """比较轮廓序列"""
        if not contour1 or not contour2:
            return 0.5

        min_len = min(len(contour1), len(contour2))
        max_len = max(len(contour1), len(contour2))

        contour1_trunc = contour1[:min_len]
        contour2_trunc = contour2[:min_len]

        matches = sum(1 for c1, c2 in zip(contour1_trunc, contour2_trunc) if c1 == c2)
        base_similarity = matches / min_len if min_len > 0 else 0

        length_penalty = min_len / max_len

        return base_similarity * length_penalty

    def _extract_rhythmic_groove_features(self, notes):
        """提取节奏律动特征"""
        if not notes:
            return {'density_pattern': [], 'syncopation': 0, 'regularity': 0.5, 'note_density': 0}

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        start_time = sorted_notes[0]['start']
        total_duration = sorted_notes[-1]['end'] - start_time

        window_size = 0.5
        num_windows = int(np.ceil(total_duration / window_size))

        density = np.zeros(num_windows)
        for note in sorted_notes:
            window_idx = int((note['start'] - start_time) / window_size)
            if 0 <= window_idx < num_windows:
                density[window_idx] += 1

        # 二值化密度模式
        density_mean = np.mean(density)
        density_pattern = [1 if d > density_mean else 0 for d in density]

        # 切分音分析
        beat_positions = np.arange(0, total_duration, 0.5)
        on_beats = 0
        off_beats = 0

        for note in sorted_notes:
            time_from_start = note['start'] - start_time
            closest_beat = round(time_from_start * 2) / 2

            if abs(time_from_start - closest_beat) < 0.05:
                on_beats += 1
            else:
                off_beats += 1

        syncopation = off_beats / (on_beats + off_beats) if (on_beats + off_beats) > 0 else 0

        # 节奏规律性
        if len(density) > 1:
            density_diff = np.abs(np.diff(density))
            regularity = 1.0 / (1.0 + np.mean(density_diff))
        else:
            regularity = 0.5

        note_density = len(notes) / total_duration if total_duration > 0 else 0

        return {
            'density_pattern': density_pattern,
            'syncopation': syncopation,
            'regularity': regularity,
            'note_density': note_density
        }

    def _compare_rhythmic_groove(self, feat1, feat2):
        """比较节奏律动"""
        # 密度模式相似度
        pattern_sim = self._compare_binary_patterns(feat1['density_pattern'], feat2['density_pattern'])

        # 切分音相似度
        syncopation_sim = 1.0 - abs(feat1['syncopation'] - feat2['syncopation'])

        # 规律性相似度
        regularity_sim = 1.0 - abs(feat1['regularity'] - feat2['regularity'])

        # 音符密度相似度
        if feat1['note_density'] > 0 and feat2['note_density'] > 0:
            density_ratio = min(feat1['note_density'], feat2['note_density']) / \
                            max(feat1['note_density'], feat2['note_density'])
        else:
            density_ratio = 0.5

        return pattern_sim * 0.4 + syncopation_sim * 0.3 + regularity_sim * 0.2 + density_ratio * 0.1

    def _compare_binary_patterns(self, pattern1, pattern2):
        """比较二值模式"""
        if not pattern1 or not pattern2:
            return 0.5

        min_len = min(len(pattern1), len(pattern2))
        pattern1_trunc = pattern1[:min_len]
        pattern2_trunc = pattern2[:min_len]

        matches = sum(1 for p1, p2 in zip(pattern1_trunc, pattern2_trunc) if p1 == p2)
        return matches / min_len if min_len > 0 else 0

    def _extract_harmonic_flavor_features(self, notes):
        """提取和声风味特征"""
        if len(notes) < 10:
            return {'chord_progression': [], 'harmonic_rhythm': 0, 'tension_profile': []}

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        start_time = sorted_notes[0]['start']
        total_duration = sorted_notes[-1]['end'] - start_time

        window_size = 1.0
        num_windows = int(np.ceil(total_duration / window_size))

        harmonic_features = []
        for w in range(num_windows):
            window_start = start_time + w * window_size
            window_end = window_start + window_size

            window_notes = [
                note for note in sorted_notes
                if window_start <= note['start'] < window_end
            ]

            if window_notes:
                pitch_classes = [note['pitch'] % 12 for note in window_notes]
                unique_pitches = list(set(pitch_classes))
                if len(unique_pitches) >= 3:
                    chord_tuple = tuple(sorted(unique_pitches[:3]))
                    harmonic_features.append(chord_tuple)
                else:
                    harmonic_features.append(None)
            else:
                harmonic_features.append(None)

        # 和声节奏
        chord_changes = 0
        for i in range(1, len(harmonic_features)):
            if harmonic_features[i] and harmonic_features[i - 1]:
                if harmonic_features[i] != harmonic_features[i - 1]:
                    chord_changes += 1

        harmonic_rhythm = chord_changes / num_windows if num_windows > 0 else 0

        # 紧张度轮廓（简化）
        tension_profile = []
        for w in range(min(10, num_windows)):  # 只取前10个窗口
            window_start = start_time + w * window_size
            window_end = window_start + window_size

            window_notes = [
                note for note in sorted_notes
                if window_start <= note['start'] < window_end
            ]

            if window_notes:
                high_notes = sum(1 for note in window_notes if note['pitch'] > 60)
                tension = high_notes / len(window_notes)
                tension_profile.append(tension)
            else:
                tension_profile.append(0)

        return {
            'chord_progression': harmonic_features,
            'harmonic_rhythm': harmonic_rhythm,
            'tension_profile': tension_profile
        }

    def _compare_harmonic_flavor(self, feat1, feat2):
        """比较和声风味"""
        # 和弦进行相似度
        progression_sim = self._compare_chord_sequences(feat1['chord_progression'], feat2['chord_progression'])

        # 和声节奏相似度
        rhythm_sim = 1.0 - abs(feat1['harmonic_rhythm'] - feat2['harmonic_rhythm'])

        # 紧张度轮廓相似度
        tension_sim = self._compare_tension_profiles(feat1['tension_profile'], feat2['tension_profile'])

        return progression_sim * 0.5 + rhythm_sim * 0.3 + tension_sim * 0.2

    def _compare_chord_sequences(self, seq1, seq2):
        """比较和弦序列"""
        if not seq1 or not seq2:
            return 0.5

        changes1 = self._extract_chord_changes(seq1)
        changes2 = self._extract_chord_changes(seq2)

        min_len = min(len(changes1), len(changes2))
        if min_len == 0:
            return 0.5

        matches = sum(1 for i in range(min_len) if changes1[i] == changes2[i])
        return matches / min_len

    def _extract_chord_changes(self, chord_sequence):
        """提取和弦变化模式"""
        changes = []
        for i in range(1, len(chord_sequence)):
            if chord_sequence[i] and chord_sequence[i - 1]:
                changes.append(0 if chord_sequence[i] == chord_sequence[i - 1] else 1)
            else:
                changes.append(0)

        return changes

    def _compare_tension_profiles(self, profile1, profile2):
        """比较紧张度轮廓"""
        if not profile1 or not profile2:
            return 0.5

        min_len = min(len(profile1), len(profile2))
        if min_len == 0:
            return 0.5

        profile1_trunc = profile1[:min_len]
        profile2_trunc = profile2[:min_len]

        # 计算相关性
        if np.std(profile1_trunc) == 0 or np.std(profile2_trunc) == 0:
            return 0.5

        corr, _ = pearsonr(profile1_trunc, profile2_trunc)
        return (corr + 1) / 2

    def _extract_structural_pattern_features(self, notes):
        """提取结构模式特征"""
        # 简化实现：返回音符数量和时间跨度
        if not notes:
            return {'note_count': 0, 'time_span': 0, 'density_variation': 0}

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        start_time = sorted_notes[0]['start']
        end_time = sorted_notes[-1]['end']
        time_span = end_time - start_time

        # 密度变化
        if len(notes) > 1 and time_span > 0:
            density_variation = np.std([note['duration'] for note in notes]) / time_span
        else:
            density_variation = 0

        return {
            'note_count': len(notes),
            'time_span': time_span,
            'density_variation': density_variation
        }

    def _compare_structural_pattern(self, feat1, feat2):
        """比较结构模式"""
        # 音符数量相似度
        if feat1['note_count'] > 0 and feat2['note_count'] > 0:
            count_ratio = min(feat1['note_count'], feat2['note_count']) / \
                          max(feat1['note_count'], feat2['note_count'])
        else:
            count_ratio = 0.5

        # 时间跨度相似度
        if feat1['time_span'] > 0 and feat2['time_span'] > 0:
            time_ratio = min(feat1['time_span'], feat2['time_span']) / \
                         max(feat1['time_span'], feat2['time_span'])
        else:
            time_ratio = 0.5

        # 密度变化相似度
        density_sim = 1.0 - abs(feat1['density_variation'] - feat2['density_variation'])

        return count_ratio * 0.4 + time_ratio * 0.3 + density_sim * 0.3

    def _extract_expressive_intensity_features(self, notes):
        """提取表达强度特征"""
        if not notes:
            return {'velocity_mean': 0, 'velocity_variance': 0, 'dynamic_range': 0}

        velocities = [note['velocity'] for note in notes]

        return {
            'velocity_mean': np.mean(velocities) if velocities else 0,
            'velocity_variance': np.var(velocities) if velocities else 0,
            'dynamic_range': max(velocities) - min(velocities) if velocities else 0
        }

    def _compare_expressive_intensity(self, feat1, feat2):
        """比较表达强度"""
        # 力度均值相似度
        if feat1['velocity_mean'] > 0 and feat2['velocity_mean'] > 0:
            mean_ratio = min(feat1['velocity_mean'], feat2['velocity_mean']) / \
                         max(feat1['velocity_mean'], feat2['velocity_mean'])
        else:
            mean_ratio = 0.5

        # 力度方差相似度
        var_sim = 1.0 - min(1.0, abs(feat1['velocity_variance'] - feat2['velocity_variance']) / 1000)

        # 动态范围相似度
        range_sim = 1.0 - min(1.0, abs(feat1['dynamic_range'] - feat2['dynamic_range']) / 127)

        return mean_ratio * 0.4 + var_sim * 0.3 + range_sim * 0.3

    # ================== 旋律轮廓提取 ==================

    def _extract_melody_contour(self, midi_obj, window_size=0.1):
        """提取旋律轮廓时间序列"""
        notes = self._extract_all_notes(midi_obj)

        if not notes:
            return np.array([0])

        start_time = min(note['start'] for note in notes)
        end_time = max(note['end'] for note in notes)

        num_windows = int(np.ceil((end_time - start_time) / window_size))
        if num_windows == 0:
            return np.array([0])

        window_pitches = np.zeros(num_windows)
        window_weights = np.zeros(num_windows)

        for note in notes:
            start_idx = int((note['start'] - start_time) / window_size)
            end_idx = int((note['end'] - start_time) / window_size)

            start_idx = max(0, min(start_idx, num_windows - 1))
            end_idx = max(0, min(end_idx, num_windows - 1))

            for idx in range(start_idx, end_idx + 1):
                window_pitches[idx] += note['pitch'] * note['duration']
                window_weights[idx] += note['duration']

        contour = np.zeros(num_windows)
        for i in range(num_windows):
            if window_weights[i] > 0:
                contour[i] = window_pitches[i] / window_weights[i]

        # 归一化
        if contour.max() >= contour.min():
            contour = (contour - contour.min()) / (contour.max() - contour.min() + 1e-8)

        # 确保返回一维数组
        return contour

    # ================== 工具方法 ==================

    def clear_cache(self):
        """清空特征缓存"""
        self.cache.clear()

    def get_cache_info(self):
        """获取缓存信息"""
        return {
            'cache_size': len(self.cache),
            'methods_available': list(self.preset_weights.keys())
        }

    def set_method(self, method):
        """设置比较方法"""
        if method in self.preset_weights:
            self.method = method
            self.weights = self.preset_weights[method]
            return True
        else:
            print(f"方法 '{method}' 不存在，可用方法: {list(self.preset_weights.keys())}")
            return False


# ================== 简单使用示例 ==================
def midi_to_pretty_midi(mido_midi, byte=False):
    """
    将mido MIDI对象转换为pretty_midi对象

    Args:
        mido_midi: mido.MidiFile对象

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    # 方法1: 通过临时文件（简单但效率较低）
    # import tempfile
    # with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
    #     mido_midi.save(tmp.name)
    #     pm = pretty_midi.PrettyMIDI(tmp.name)
    # return pm

    # 方法2: 通过字节流（更高效）
    # 创建一个字节流来保存MIDI数据
    import io
    midi_bytes = io.BytesIO()
    mido_midi.save(file=midi_bytes)
    midi_bytes.seek(0)  # 回到起始位置

    # 使用pretty_midi从字节流加载
    pm = pretty_midi.PrettyMIDI(midi_bytes)
    if byte:
        return pm, midi_bytes
    return pm

def create_sample_midi():
    """创建示例MIDI文件用于测试"""
    import tempfile

    # 创建第一个MIDI
    midi1 = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # 添加C大调音阶
    start_time = 0
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_time + i * 0.5,
            end=start_time + i * 0.5 + 0.4
        )
        piano.notes.append(note)

    midi1.instruments.append(piano)

    # 创建第二个MIDI（相似但不完全相同）
    midi2 = pretty_midi.PrettyMIDI()
    piano2 = pretty_midi.Instrument(program=piano_program)

    # 添加稍有变化的音阶
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=90,
            pitch=pitch + 1,  # 移调一个半音
            start=start_time + i * 0.55,  # 节奏稍有变化
            end=start_time + i * 0.55 + 0.35
        )
        piano2.notes.append(note)

    midi2.instruments.append(piano2)

    # 保存到临时文件
    temp_dir = tempfile.gettempdir()
    midi1_path = os.path.join(temp_dir, 'sample1.mid')
    midi2_path = os.path.join(temp_dir, 'sample2.mid')

    midi1.write(midi1_path)
    midi2.write(midi2_path)

    return midi1_path, midi2_path