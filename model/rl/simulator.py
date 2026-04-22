import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json

from torch.functional import Tensor

from utils.GP5Generator import GuitarTechnique


# ====================== 配置类 ======================

@dataclass
class DifficultyConfig:
    """难度配置类 - 所有参数都可以在这里调整"""

    # 基础设置
    bpm: int = 120
    non_playing_fret: int = 25
    measure_length: int = 16  # 每小节16个十六分音符

    # 位置难度参数
    position_base: float = 0.0  # 基础位置难度

    # 品位相关参数
    fret_difficulty_enabled: bool = True
    fret_difficulty_base: float = 0.5  # 每品基础难度
    fret_difficulty_curve: str = "quadratic"  # 可选: linear, quadratic, exponential
    low_fret_limit: int = 5  # 低于此品位的难度降低
    high_fret_penalty_start: int = 12  # 从此品位开始增加难度
    high_fret_penalty_factor: float = 1.5

    # 手指相关参数
    finger_stretch_base: float = 0.0
    finger_stretch_weight: float = 2.0
    max_comfortable_stretch: int = 4  # 舒适伸展范围（品）

    # 音符数量相关参数
    multi_note_threshold: int = 3  # 超过此数量算多音符
    single_note_base: float = 1.0
    multi_note_base: float = 3.0
    chord_complexity_factor: float = 1.2  # 和弦复杂度因子

    # 移动难度参数
    move_base_difficulty: float = 0.0
    move_distance_weight: float = 2.0
    move_pattern_weight: float = 3.0

    # 单音符/多音符移动差异化
    single_note_move_factor: float = 0.5  # 单音符移动难度乘数
    multi_note_move_factor: float = 2.0  # 多音符移动难度乘数
    mixed_note_move_factor: float = 1.5  # 混合音符移动难度乘数

    # 转换相关参数
    transition_time_pressure: bool = True
    time_pressure_threshold: float = 1.2  # 时间压力阈值（相对于十六分音符）
    time_pressure_weight: float = 3.0

    # 连续难度参数
    accumulated_difficulty_enabled: bool = True
    accumulated_weight: float = 0.1
    stamina_drain_factor: float = 0.01

    # 跨行/跨小节参数
    measure_boundary_penalty: float = 2.0
    line_break_penalty: float = 3.0

    # 难度等级阈值
    difficulty_thresholds: Dict[str, float] = None

    # 自定义难度计算公式
    custom_difficulty_formula: Optional[Callable] = None

    def __post_init__(self):
        if self.difficulty_thresholds is None:
            self.difficulty_thresholds = {
                "beginner": 10,
                "easy": 30,
                "intermediate": 70,
                "advanced": 150,
                "expert": 300,
                "virtuoso": 500
            }

    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.__dict__.copy()

    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'DifficultyConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class DifficultyLevel(Enum):
    """难度等级"""
    BEGINNER = "初学者"
    EASY = "简单"
    INTERMEDIATE = "中等"
    ADVANCED = "高级"
    EXPERT = "专家级"
    VIRTUOSO = "大师级"

    @classmethod
    def from_score(cls, score: float, thresholds: Dict[str, float]) -> 'DifficultyLevel':
        """根据分数确定难度等级"""
        if score < thresholds.get("beginner", 10):
            return cls.BEGINNER
        elif score < thresholds.get("easy", 30):
            return cls.EASY
        elif score < thresholds.get("intermediate", 70):
            return cls.INTERMEDIATE
        elif score < thresholds.get("advanced", 150):
            return cls.ADVANCED
        elif score < thresholds.get("expert", 300):
            return cls.EXPERT
        else:
            return cls.VIRTUOSO


# ====================== 手部模型 ======================

class HandModel:
    """手部物理模型"""

    def __init__(self, config: DifficultyConfig):
        self.config = config

        # 手指参数
        self.fingers = {
            1: {"name": "食指", "strength": 1.0, "flexibility": 0.8, "span": 4.0},
            2: {"name": "中指", "strength": 0.9, "flexibility": 0.9, "span": 2.5},
            3: {"name": "无名指", "strength": 0.8, "flexibility": 0.8, "span": 2.0},
            4: {"name": "小指", "strength": 0.6, "flexibility": 0.6, "span": 2.0}
        }

    def calculate_fret_difficulty(self, fret: int) -> float:
        """计算单品位难度"""
        if not self.config.fret_difficulty_enabled or fret <= 0:
            return 0.0

        # 底品难度降低或取消
        if fret <= self.config.low_fret_limit:
            base = self.config.fret_difficulty_base * 0.3
        elif fret <= self.config.high_fret_penalty_start:
            base = self.config.fret_difficulty_base
        else:
            base = self.config.fret_difficulty_base * self.config.high_fret_penalty_factor

        # 难度曲线
        if self.config.fret_difficulty_curve == "quadratic":
            difficulty = base * (fret ** 1.5)
        elif self.config.fret_difficulty_curve == "exponential":
            difficulty = base * (1.5 ** (fret / 5))
        else:  # linear
            difficulty = base * fret

        return difficulty

    def calculate_stretch_difficulty(self, frets: List[int]) -> float:
        """计算伸展难度"""
        if len(frets) < 2:
            return 0.0

        min_fret = min(frets)
        max_fret = max(frets)
        stretch = max_fret - min_fret

        if stretch <= self.config.max_comfortable_stretch:
            return 0.0

        excess_stretch = stretch - self.config.max_comfortable_stretch
        return excess_stretch * self.config.finger_stretch_weight

    def calculate_note_count_difficulty(self, note_count: int) -> float:
        """计算音符数量难度"""
        if note_count == 0:
            return 0.0
        elif note_count == 1:
            return self.config.single_note_base
        elif note_count <= self.config.multi_note_threshold:
            return self.config.multi_note_base
        else:
            # 超过阈值的音符数量额外增加难度
            excess = note_count - self.config.multi_note_threshold
            return self.config.multi_note_base * (self.config.chord_complexity_factor ** excess)


# ====================== 位置分析 ======================

class PositionAnalyzer:
    """位置分析器 - 优化版：保留原接口，内部使用向量化计算（批量）"""

    def __init__(self, config: DifficultyConfig, hand_model: HandModel):
        self.config = config
        self.hand_model = hand_model
        self.technique_difficulty_factors = {
            GuitarTechnique.NORMAL: 1.0,
            GuitarTechnique.HAMMER_ON: 1.2,
            GuitarTechnique.PULL_OFF: 1.2,
            GuitarTechnique.SLIDE: 1.3,
            GuitarTechnique.BEND: 1.5,
            GuitarTechnique.VIBRATO: 1.4,
            GuitarTechnique.MUTE: 0.8,
            GuitarTechnique.NATURAL_HARMONIC: 1.3,
            GuitarTechnique.ARTIFICIAL_HARMONIC: 1.6,
            GuitarTechnique.TAPPED_HARMONIC: 1.7,
            GuitarTechnique.PINCH_HARMONIC: 1.8,
            GuitarTechnique.SEMI_HARMONIC: 1.4,
            GuitarTechnique.TREMOLO: 1.5,
            GuitarTechnique.PALM_MUTE: 0.9,
        }

    # 原有单个位置分析方法（保留，用于向后兼容，但内部可调用向量化）
    def analyze_position(self, frets: List[int], techniques: List[GuitarTechnique] = None) -> Dict[str, Any]:
        """单个位置分析（保持原接口，内部通过向量化实现）"""
        # 为了兼容性，简单调用批量方法再提取
        batch_result = self.analyze_positions_batch([frets], [techniques] if techniques else None)
        return {k: v[0] for k, v in batch_result.items() if isinstance(v, (list, np.ndarray))}

    def analyze_positions_batch(self, frets_list: List[List[int]],
                                techniques_list: Optional[List[List[GuitarTechnique]]] = None) -> Dict[str, Any]:
        """
        批量分析多个位置（向量化核心）
        返回字典，每个值均为数组，长度 = len(frets_list)
        """
        T = len(frets_list)
        if techniques_list is None:
            techniques_list = [[GuitarTechnique.NORMAL] * 6] * T

        # 转换为 numpy 数组
        frets = np.array(frets_list, dtype=np.int8)                     # (T, 6)
        tech_ints = np.array([[t.value for t in row] for row in techniques_list], dtype=np.int8)  # (T,6)

        # 有效音符掩码
        valid_mask = (frets > 0) & (frets < self.config.non_playing_fret)  # (T,6)
        note_count = valid_mask.sum(axis=1).astype(np.float32)             # (T,)

        # ----- 品位难度（向量化）-----
        # 预先计算品位难度系数
        fret_difficulty_raw = self._calc_fret_difficulty_batch(frets)      # (T,6)
        fret_difficulty = np.where(valid_mask, fret_difficulty_raw, 0.0)
        avg_fret_difficulty = np.divide(fret_difficulty.sum(axis=1), note_count,
                                        out=np.zeros_like(note_count), where=note_count>0)

        # ----- 伸展难度（向量化）-----
        # 有效品位的最小值和最大值
        min_fret = np.where(valid_mask, frets, 999).min(axis=1)
        max_fret = np.where(valid_mask, frets, -1).max(axis=1)
        stretch = max_fret - min_fret
        stretch_difficulty = np.maximum(0, stretch - self.config.max_comfortable_stretch) * self.config.finger_stretch_weight
        stretch_difficulty[note_count < 2] = 0.0

        # ----- 音符数量难度（向量化）-----
        note_count_difficulty = np.zeros_like(note_count)
        single_mask = note_count == 1
        multi_mask = (note_count > 1) & (note_count <= self.config.multi_note_threshold)
        excess_mask = note_count > self.config.multi_note_threshold

        note_count_difficulty[single_mask] = self.config.single_note_base
        note_count_difficulty[multi_mask] = self.config.multi_note_base
        # 超过阈值部分
        excess = note_count[excess_mask] - self.config.multi_note_threshold
        note_count_difficulty[excess_mask] = self.config.multi_note_base * (self.config.chord_complexity_factor ** excess)

        # ----- 技巧难度（向量化）-----
        # 获取每个技巧的难度因子
        tech_factor = np.vectorize(lambda t: self.technique_difficulty_factors.get(GuitarTechnique(t), 1.0),
                                   otypes=[np.float32])(tech_ints)   # (T,6)
        # 高品位惩罚
        high_fret_mask = (frets > 12) & np.isin(tech_ints,
                                                [GuitarTechnique.BEND.value, GuitarTechnique.VIBRATO.value])
        tech_factor = np.where(high_fret_mask,
                               tech_factor * (1 + (frets - 12) * 0.02),
                               tech_factor)
        tech_difficulty = (tech_factor * valid_mask).sum(axis=1)  # (T,)

        # 多复杂技巧组合惩罚
        complex_tech_vals = [t.value for t in [GuitarTechnique.NORMAL, GuitarTechnique.MUTE, GuitarTechnique.PALM_MUTE]]
        is_complex = ~np.isin(tech_ints, complex_tech_vals)       # (T,6)
        complex_count = (is_complex & valid_mask).sum(axis=1)
        combo_factor = np.where(complex_count > 1, 1 + (complex_count - 1) * 0.1, 1.0)
        tech_difficulty *= combo_factor

        # 总难度
        base_diff = self.config.position_base
        total_difficulty = (base_diff + avg_fret_difficulty + stretch_difficulty +
                            note_count_difficulty + tech_difficulty)

        # 辅助信息：激活弦编号集合（用位掩码表示，便于后续移动分析）
        string_indices = np.arange(1, 7)  # 1..6
        active_strings_mask = valid_mask * string_indices  # (T,6) 有效位置存弦号，无效为0
        # 编码为整数集合（位掩码）：第i位表示弦i是否激活
        string_sets = (valid_mask << np.arange(6)).sum(axis=1)  # (T,) 整数掩码

        return {
            "total_difficulty": total_difficulty,           # (T,)
            "note_count": note_count,                       # (T,)
            "active_strings_mask": string_sets,             # (T,)
            "frets": frets,                                 # (T,6)
            "tech_ints": tech_ints,                         # (T,6)
            "valid_mask": valid_mask,                       # (T,6)
            "avg_fret_difficulty": avg_fret_difficulty,
            "stretch_difficulty": stretch_difficulty,
            "note_count_difficulty": note_count_difficulty,
            "tech_difficulty": tech_difficulty,
        }

    def _calc_fret_difficulty_batch(self, frets: np.ndarray) -> np.ndarray:
        """批量计算品位难度矩阵 (T,6) - 修复无效值警告"""
        config = self.config
        # 创建副本，将无效品位（>= non_playing_fret）临时设为0，避免 power 计算产生 inf
        frets_safe = np.where(frets < config.non_playing_fret, frets, 0)

        low_mask = frets_safe <= config.low_fret_limit
        mid_mask = (frets_safe > config.low_fret_limit) & (frets_safe <= config.high_fret_penalty_start)
        high_mask = frets_safe > config.high_fret_penalty_start

        base = np.zeros_like(frets_safe, dtype=np.float32)
        base[low_mask] = config.fret_difficulty_base * 0.3
        base[mid_mask] = config.fret_difficulty_base
        base[high_mask] = config.fret_difficulty_base * config.high_fret_penalty_factor

        if config.fret_difficulty_curve == "quadratic":
            return base * (frets_safe ** 1.5)
        elif config.fret_difficulty_curve == "exponential":
            return base * (1.5 ** (frets_safe / 5))
        else:  # linear
            return base * frets_safe


# ====================== 移动分析 ======================

class MoveAnalyzer:
    """移动分析器 - 优化版：支持批量计算"""

    def __init__(self, config: DifficultyConfig):
        self.config = config

    # 保留原单对移动方法（兼容旧调用）
    def calculate_move_difficulty(self, pos1_info: Dict, pos2_info: Dict, time_interval: float) -> Dict[str, Any]:
        # 简单调用批量方法并提取第一个元素（为保持兼容，不推荐直接调用）
        batch_res = self.calculate_moves_batch([pos1_info], [pos2_info], [time_interval])
        return {k: v[0] for k, v in batch_res.items()}

    def _bit_count(self, arr: np.ndarray) -> np.ndarray:
        """向量化计算整数数组的位计数（每个元素二进制中1的个数）"""
        # 如果 NumPy 版本足够，直接用 bitwise_count
        if hasattr(np, 'bitwise_count'):
            return np.bitwise_count(arr)
        else:
            # 手动向量化：使用列表推导（效率稍低，但弦数少可接受）
            return np.array([bin(x).count('1') for x in arr])

    def calculate_moves_batch(self, pos1_list: List[Dict], pos2_list: List[Dict],
                              time_intervals: List[float]) -> Dict[str, np.ndarray]:
        """
        批量计算移动难度 - 修复 bit_count 错误
        """
        M = len(pos1_list)
        if M == 0:
            return {
                "total_difficulty": np.array([]),
                "distance_difficulty": np.array([]),
                "pattern_difficulty": np.array([]),
                "time_pressure": np.array([]),
                "move_factor": np.array([]),
            }

        note_count1 = np.array([p["note_count"] for p in pos1_list])
        note_count2 = np.array([p["note_count"] for p in pos2_list])
        string_sets1 = np.array([p.get("active_strings_mask", 0) for p in pos1_list])
        string_sets2 = np.array([p.get("active_strings_mask", 0) for p in pos2_list])
        centers1 = np.array([p.get("center", (0.0, 0.0)) for p in pos1_list])  # (M,2)
        centers2 = np.array([p.get("center", (0.0, 0.0)) for p in pos2_list])
        time_intervals = np.array(time_intervals)

        # 移动因子
        single_note = (note_count1 == 1) & (note_count2 == 1)
        multi_note = (note_count1 >= self.config.multi_note_threshold) & (
                    note_count2 >= self.config.multi_note_threshold)
        move_factor = np.where(single_note, self.config.single_note_move_factor,
                               np.where(multi_note, self.config.multi_note_move_factor,
                                        self.config.mixed_note_move_factor))

        # 距离难度
        delta = centers2 - centers1
        distances = np.sqrt(delta[:, 0] ** 2 + (delta[:, 1] * 2) ** 2)
        distance_difficulty = distances * self.config.move_distance_weight * move_factor

        # 模式变化难度 - 使用位计数
        intersection = self._bit_count(string_sets1 & string_sets2)
        union = self._bit_count(string_sets1 | string_sets2)
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        pattern_change = (1 - similarity) * self.config.move_pattern_weight
        pattern_change += np.abs(note_count1 - note_count2) * 0.2
        pattern_difficulty = pattern_change * move_factor

        # 时间压力
        sixteenth_duration = 60 / self.config.bpm / 4
        threshold = sixteenth_duration * self.config.time_pressure_threshold
        pressure = np.maximum(0, (threshold - time_intervals) / sixteenth_duration)
        time_pressure = pressure * self.config.time_pressure_weight

        total = (self.config.move_base_difficulty + distance_difficulty +
                 pattern_difficulty + time_pressure)

        return {
            "total_difficulty": total,
            "distance_difficulty": distance_difficulty,
            "pattern_difficulty": pattern_difficulty,
            "time_pressure": time_pressure,
            "move_factor": move_factor,
        }


# ====================== 序列分析 ======================

class GuitarSequenceAnalyzer:
    """序列分析器"""

    def __init__(self, config: DifficultyConfig):
        self.config = config
        self.hand_model = HandModel(config)
        self.position_analyzer = PositionAnalyzer(config, self.hand_model)
        self.move_analyzer = MoveAnalyzer(config)

    def evaluate(self, sequence_data) -> float:
        """奖励函数入口：返回总体难度分数"""
        report = self.analyze_sequence(sequence_data)
        return report["statistics"]["overall_difficulty"]

    def parse_sequence(self, sequence_data) -> List[Dict]:
        """解析序列数据 - 支持新的字典格式和旧格式"""
        parsed = []

        # 判断输入格式
        if isinstance(sequence_data, dict) and 'fret' in sequence_data:
            # 新的字典格式：包含整个序列的fret和technique
            frets_data = sequence_data['fret']
            techniques_data = sequence_data.get('technique')

            # 转换为列表格式（如果是张量）
            if isinstance(frets_data, Tensor):
                frets_list = frets_data.tolist()
            elif isinstance(frets_data, list):
                frets_list = frets_data
            else:
                raise ValueError(f"fret数据格式不支持: {type(frets_data)}")

            # 检查technique数据
            if techniques_data is None:
                # 如果没有提供technique，使用默认的NORMAL技巧
                techniques_list = [[GuitarTechnique.NORMAL.value] * 6 for _ in range(len(frets_list))]
            else:
                # 处理technique数据
                if isinstance(techniques_data, Tensor):
                    techniques_list = techniques_data.tolist()
                elif isinstance(techniques_data, list):
                    techniques_list = techniques_data
                else:
                    raise ValueError(f"technique数据格式不支持: {type(techniques_data)}")

                # 检查形状匹配
                if len(techniques_list) != len(frets_list):
                    raise ValueError(f"fret和technique的长度不匹配: {len(frets_list)} vs {len(techniques_list)}")

                # 处理technique的形状
                for i in range(len(techniques_list)):
                    tech = techniques_list[i]
                    if isinstance(tech, int):
                        # 单个int：所有弦使用同一技巧
                        techniques_list[i] = [tech] * 6
                    elif isinstance(tech, list) and len(tech) == 6:
                        # 已经是长度为6的列表
                        pass
                    else:
                        raise ValueError(f"第{i + 1}个和弦technique格式错误: {tech}")

            # 遍历每个和弦
            for i in range(len(frets_list)):
                frets = frets_list[i]
                tech_ints = techniques_list[i]

                # 验证frets长度
                if len(frets) != 6:
                    raise ValueError(f"第{i + 1}个和弦fret长度应为6，实际为{len(frets)}: {frets}")

                # 将int技巧映射为GuitarTechnique枚举值
                techniques = [self._map_int_to_technique(tech_int) for tech_int in tech_ints]

                # 时间位置
                sixteenth_duration = 60 / self.config.bpm / 4
                time_position = i * sixteenth_duration
                measure_num = i // self.config.measure_length
                measure_pos = i % self.config.measure_length

                parsed.append({
                    "index": i,
                    "time": time_position,
                    "frets": frets,
                    "techniques": techniques,
                    "measure_num": measure_num,
                    "measure_pos": measure_pos,
                    "is_line_break": measure_pos == self.config.measure_length - 1 and i < len(frets_list) - 1
                })

        elif isinstance(sequence_data, (list, tuple)):
            # 旧格式：和弦列表
            for i, chord_data in enumerate(sequence_data):
                frets = []
                techniques = []

                if isinstance(chord_data, (list, tuple)):
                    # 旧格式：纯列表，只有fret
                    if len(chord_data) == 6:
                        frets = list(chord_data)
                        techniques = [GuitarTechnique.NORMAL] * 6
                    elif len(chord_data) == 2 and isinstance(chord_data[0], (list, tuple)) and len(chord_data[0]) == 6:
                        # 兼容格式：[fret_list, technique_list]
                        frets = list(chord_data[0])
                        tech_input = chord_data[1]

                        if isinstance(tech_input, int):
                            # 单个int技巧
                            tech_value = self._map_int_to_technique(tech_input)
                            techniques = [tech_value] * 6
                        elif isinstance(tech_input, (list, tuple)) and len(tech_input) == 6:
                            # 每弦一个技巧
                            techniques = [self._map_int_to_technique(tech_int) for tech_int in tech_input]
                        else:
                            raise ValueError(f"第{i + 1}个和弦技巧格式错误: {chord_data}")
                    else:
                        raise ValueError(f"第{i + 1}个和弦格式错误: {chord_data}")
                else:
                    raise ValueError(f"第{i + 1}个和弦格式错误: {chord_data}")

                # 验证frets长度
                if len(frets) != 6:
                    raise ValueError(f"第{i + 1}个和弦fret长度应为6，实际为{len(frets)}: {frets}")

                # 验证techniques长度
                if len(techniques) != 6:
                    raise ValueError(f"第{i + 1}个和弦technique长度应为6，实际为{len(techniques)}: {techniques}")

                # 时间位置
                sixteenth_duration = 60 / self.config.bpm / 4
                time_position = i * sixteenth_duration
                measure_num = i // self.config.measure_length
                measure_pos = i % self.config.measure_length

                parsed.append({
                    "index": i,
                    "time": time_position,
                    "frets": frets,
                    "techniques": techniques,
                    "measure_num": measure_num,
                    "measure_pos": measure_pos,
                    "is_line_break": measure_pos == self.config.measure_length - 1 and i < len(sequence_data) - 1
                })
        else:
            raise ValueError(f"序列数据格式不支持: {type(sequence_data)}")

        return parsed

    def _map_int_to_technique(self, tech_int: int) -> GuitarTechnique:
        """将int映射为GuitarTechnique枚举值"""
        try:
            # 尝试直接通过值获取枚举
            return GuitarTechnique(tech_int)
        except ValueError:
            # 如果值不在枚举定义中，记录警告并返回NORMAL
            print(f"警告: 未知的技巧值 {tech_int}，将使用 NORMAL 替代")
            return GuitarTechnique.NORMAL

    def analyze_sequence(self, sequence_data) -> Dict[str, Any]:
        """完全向量化版本的分析函数"""
        # 1. 解析序列
        sequence = self.parse_sequence(sequence_data)   # 返回列表，每个元素是 dict
        if not sequence:
            return self._empty_report()

        T = len(sequence)
        # 提取所有位置的 frets 和 techniques
        frets_list = [item["frets"] for item in sequence]
        techs_list = [item["techniques"] for item in sequence]

        # 2. 批量计算位置难度及中间信息
        pos_batch = self.position_analyzer.analyze_positions_batch(frets_list, techs_list)
        pos_diffs = pos_batch["total_difficulty"]          # (T,)
        note_counts = pos_batch["note_count"]              # (T,)
        string_sets = pos_batch["active_strings_mask"]     # (T,)
        frets_arr = pos_batch["frets"]                     # (T,6)
        valid_mask = pos_batch["valid_mask"]               # (T,6)

        # 计算每个位置的中心点（向量化）
        centers = self._compute_centers_batch(frets_arr, valid_mask)   # (T,2)

        # 构建每个位置的完整信息字典（用于后续报告，但难度值已用向量化结果覆盖）
        positions = []
        for i in range(T):
            pos_info = {
                "index": sequence[i]["index"],
                "time": sequence[i]["time"],
                "measure_num": sequence[i]["measure_num"],
                "is_line_break": sequence[i]["is_line_break"],
                "techniques": sequence[i]["techniques"],
                "frets": frets_list[i],
                "active_frets": [(s+1, frets_arr[i,s]) for s in range(6) if valid_mask[i,s]],
                "active_fret_values": [frets_arr[i,s] for s in range(6) if valid_mask[i,s]],
                "active_strings": [s+1 for s in range(6) if valid_mask[i,s]],
                "note_count": int(note_counts[i]),
                "is_single_note": note_counts[i] == 1,
                "is_multi_note": note_counts[i] >= self.config.multi_note_threshold,
                "total_difficulty": float(pos_diffs[i]),
                "center": (centers[i,0], centers[i,1]),
                "active_strings_mask": int(string_sets[i]),
            }
            positions.append(pos_info)

        # 3. 批量计算移动难度（需要相邻位置）
        if T < 2:
            moves = []
            move_diffs = np.array([])
        else:
            pos1_list = positions[:-1]
            pos2_list = positions[1:]
            time_intervals = [positions[i+1]["time"] - positions[i]["time"] for i in range(T-1)]
            move_batch = self.move_analyzer.calculate_moves_batch(pos1_list, pos2_list, time_intervals)
            move_diffs = move_batch["total_difficulty"]   # (T-1,)

            # 构建移动信息列表（用于报告）
            moves = []
            for i in range(T-1):
                move_info = {
                    "from_index": i,
                    "to_index": i+1,
                    "total_difficulty": float(move_diffs[i]),
                    "components": {
                        "base_move": self.config.move_base_difficulty,
                        "distance": float(move_batch["distance_difficulty"][i]),
                        "pattern": float(move_batch["pattern_difficulty"][i]),
                        "time_pressure": float(move_batch["time_pressure"][i]),
                    },
                    "move_factor": float(move_batch["move_factor"][i]),
                    "note_counts": (int(note_counts[i]), int(note_counts[i+1])),
                    "is_cross_measure": positions[i]["measure_num"] != positions[i+1]["measure_num"],
                }
                moves.append(move_info)

        # 4. 累计难度
        if self.config.accumulated_difficulty_enabled:
            # 位置难度和移动难度交错累加
            pos_diffs_arr = pos_diffs
            move_diffs_arr = move_diffs if len(move_diffs) > 0 else np.array([])
            # 构建完整序列： [pos0, move0, pos1, move1, ...]
            interleaved = []
            for i in range(T):
                interleaved.append(pos_diffs_arr[i])
                if i < T-1:
                    interleaved.append(move_diffs_arr[i])
            interleaved = np.array(interleaved)
            accumulated_arr = np.cumsum(interleaved)
            # 每4个元素衰减一次（每两个位置一个衰减周期，保持与原逻辑一致）
            for idx in range(4, len(accumulated_arr), 4):
                accumulated_arr[idx:] *= 0.95
            # 提取每个位置结束时的累计难度
            accumulated = [float(accumulated_arr[2*i]) for i in range(T)]
        else:
            accumulated = [0.0] * T

        # 5. 统计量计算
        stats = self._compute_statistics(pos_diffs, move_diffs, accumulated, positions, moves)
        overall_difficulty = stats["overall_difficulty"]

        # 6. 确定难度等级
        difficulty_level = DifficultyLevel.from_score(overall_difficulty, self.config.difficulty_thresholds)

        # 7. 生成报告（保持与原结构相同）
        report = self._generate_report(positions, moves, overall_difficulty, difficulty_level,
                                       pos_diffs.tolist(), move_diffs.tolist(), accumulated, stats)
        return report

    def _compute_centers_batch(self, frets: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """批量计算手位中心 (T,2) : (弦, 品) - 修复除零警告"""
        T = frets.shape[0]
        strings = np.arange(1, 7, dtype=np.float32)  # 1..6
        string_weights = (6 - strings) * 0.3 + 0.7  # (6,)
        fret_weights = 1 + frets / 24.0  # (T,6)
        weights = string_weights * fret_weights  # (T,6)
        weights = weights * valid_mask  # 无效位置权重为0
        sum_weights = weights.sum(axis=1, keepdims=True)  # (T,1)

        # 使用 np.divide 避免除零警告，同时保留零位置
        center_string = np.divide((weights * strings).sum(axis=1), sum_weights.squeeze(),
                                  out=np.zeros(T, dtype=np.float32), where=sum_weights.squeeze() != 0)
        center_fret = np.divide((weights * frets).sum(axis=1), sum_weights.squeeze(),
                                out=np.zeros(T, dtype=np.float32), where=sum_weights.squeeze() != 0)
        return np.stack([center_string, center_fret], axis=1)  # (T,2)

    def _compute_statistics(self, pos_diffs: np.ndarray, move_diffs: np.ndarray,
                            accumulated: List[float], positions: List[Dict], moves: List[Dict]) -> Dict:
        """计算各种统计量（向量化后）"""
        if len(pos_diffs) == 0:
            return {
                "overall_difficulty": 0.0,
                "difficulty_level": DifficultyLevel.BEGINNER.value,
                "bpm": self.config.bpm,
                "total_positions": 0,
                "total_moves": 0,
                "final_stamina": 100.0,
                "avg_position_difficulty": 0.0,
                "max_position_difficulty": 0.0,
                "avg_move_difficulty": 0.0,
                "max_move_difficulty": 0.0,
                "max_accumulated_difficulty": 0.0,
            }

        avg_pos = np.mean(pos_diffs)
        max_pos = np.max(pos_diffs)
        avg_move = np.mean(move_diffs) if len(move_diffs) > 0 else 0.0
        max_move = np.max(move_diffs) if len(move_diffs) > 0 else 0.0
        max_accum = max(accumulated) if accumulated else 0.0

        # 耐力模拟
        stamina = 100.0
        for diff in pos_diffs:
            stamina -= diff * self.config.stamina_drain_factor
            stamina = max(0.0, stamina)

        # 多音符因子
        multi_note_count = sum(1 for p in positions if p["note_count"] >= self.config.multi_note_threshold)
        multi_note_factor = 1.0 + (multi_note_count / max(len(positions), 1)) * 0.5

        # 跨小节因子
        cross_measure_count = sum(1 for m in moves if m.get("is_cross_measure", False))
        cross_measure_factor = 1.0 + (cross_measure_count / max(len(moves), 1)) * 0.3

        # 总体难度公式
        overall = (avg_pos * 0.25 + max_pos * 0.15 +
                   avg_move * 0.30 + max_move * 0.20 +
                   max_accum * self.config.accumulated_weight) * multi_note_factor * cross_measure_factor

        # 若提供了自定义公式则使用
        if self.config.custom_difficulty_formula:
            overall = self.config.custom_difficulty_formula(
                pos_diffs.tolist(), move_diffs.tolist(), accumulated, positions, moves
            )

        return {
            "overall_difficulty": overall,
            "difficulty_level": DifficultyLevel.from_score(overall, self.config.difficulty_thresholds).value,
            "bpm": self.config.bpm,
            "total_positions": len(positions),
            "total_moves": len(moves),
            "final_stamina": stamina,
            "avg_position_difficulty": float(avg_pos),
            "max_position_difficulty": float(max_pos),
            "avg_move_difficulty": float(avg_move),
            "max_move_difficulty": float(max_move),
            "max_accumulated_difficulty": float(max_accum),
        }

    def _generate_report(self, positions, moves, overall_diff, difficulty_level,
                         pos_diffs, move_diffs, accumulated, stats) -> Dict:
        """生成详细报告（与原报告结构一致）"""
        # 最难位置
        hardest_positions = []
        if positions:
            sorted_idx = np.argsort(pos_diffs)[-3:][::-1]
            hardest_positions = [
                {"index": positions[i]["index"], "difficulty": pos_diffs[i],
                 "note_count": positions[i]["note_count"]}
                for i in sorted_idx
            ]

        # 最难移动
        hardest_moves = []
        if moves and move_diffs:
            sorted_idx = np.argsort(move_diffs)[-3:][::-1]
            hardest_moves = [
                {"from": moves[i]["from_index"], "to": moves[i]["to_index"],
                 "difficulty": move_diffs[i], "move_factor": moves[i]["move_factor"],
                 "note_counts": moves[i]["note_counts"]}
                for i in sorted_idx
            ]

        # 多音符统计
        multi_note_stats = {
            "total_multi_notes": sum(1 for p in positions if p["note_count"] >= self.config.multi_note_threshold),
            "max_notes_in_position": max((p["note_count"] for p in positions), default=0)
        }

        # 建议（复用原逻辑）
        suggestions = self._generate_suggestions(overall_diff, difficulty_level,
                                                 hardest_positions, hardest_moves, multi_note_stats)

        return {
            "statistics": stats,
            "hardest_positions": hardest_positions,
            "hardest_moves": hardest_moves,
            "multi_note_analysis": multi_note_stats,
            "suggestions": suggestions,
            "config_summary": self.config.to_dict()
        }

    def _empty_report(self):
        """空序列报告"""
        return {
            "statistics": {
                "overall_difficulty": 0.0,
                "difficulty_level": DifficultyLevel.BEGINNER.value,
                "bpm": self.config.bpm,
                "total_positions": 0,
                "total_moves": 0,
                "final_stamina": 100.0,
                "avg_position_difficulty": 0.0,
                "max_position_difficulty": 0.0,
                "avg_move_difficulty": 0.0,
                "max_move_difficulty": 0.0,
                "max_accumulated_difficulty": 0.0,
            },
            "hardest_positions": [],
            "hardest_moves": [],
            "multi_note_analysis": {"total_multi_notes": 0, "max_notes_in_position": 0},
            "suggestions": [],
            "config_summary": self.config.to_dict()
        }

    def _generate_suggestions(self, overall_diff, difficulty_level, hardest_positions,
                             hardest_moves, multi_note_stats):
        """生成练习建议"""
        suggestions = []

        # 基于总体难度
        if overall_diff > 300:
            suggestions.append("大师级难度，建议专业指导+分解练习")
            suggestions.append("每天练习不超过1小时，避免受伤")
        elif overall_diff > 150:
            suggestions.append("专家级难度，需要系统训练计划")
            suggestions.append("重点关注技术和耐力训练")
        elif overall_diff > 70:
            suggestions.append("高级难度，适合挑战性练习")
            suggestions.append("使用节拍器从慢速开始逐步加速")
        elif overall_diff > 30:
            suggestions.append("中等难度，适合日常技术提升")
            suggestions.append("注意手型和放松")
        else:
            suggestions.append("初级难度，适合基础练习")
            suggestions.append("建立正确的手型和指法习惯")

        # 多音符建议
        if multi_note_stats["total_multi_notes"] > 0:
            suggestions.append(f"包含{multi_note_stats['total_multi_notes']}个多音符位置，单独练习和弦按法")
            if multi_note_stats["max_notes_in_position"] >= 4:
                suggestions.append("有复杂和弦，练习手指独立性和力量")

        # 针对最难位置
        if hardest_positions:
            hardest = hardest_positions[0]
            if hardest["difficulty"] > 50:
                suggestions.append(f"位置{hardest['index'] + 1}难度极高，建议单独练习")

        # 针对最难移动
        if hardest_moves:
            hardest = hardest_moves[0]
            if hardest["difficulty"] > 30:
                suggestions.append(f"移动{hardest['from'] + 1}→{hardest['to'] + 1}转换困难，练习流畅转换")
                if hardest["move_factor"] > 1.5:
                    suggestions.append("多音符转换，练习手部协调性")

        return suggestions


# ====================== 预置配置 ======================

class PresetConfigs:
    """预置配置"""

    @staticmethod
    def get_default() -> DifficultyConfig:
        """默认配置"""
        return DifficultyConfig()

    @staticmethod
    def get_easy() -> DifficultyConfig:
        """简单配置"""
        config = DifficultyConfig()
        config.single_note_move_factor = 0.3
        config.multi_note_move_factor = 1.5
        config.fret_difficulty_base = 0.3
        config.fret_difficulty_curve = "linear"
        return config

    @staticmethod
    def get_hard() -> DifficultyConfig:
        """困难配置"""
        config = DifficultyConfig()
        config.single_note_move_factor = 0.7
        config.multi_note_move_factor = 2.5
        config.fret_difficulty_base = 0.8
        config.fret_difficulty_curve = "quadratic"
        config.multi_note_threshold = 2  # 更严格的多音符判断
        return config

    @staticmethod
    def get_extreme() -> DifficultyConfig:
        """极限配置"""
        config = DifficultyConfig()
        config.single_note_move_factor = 1.0
        config.multi_note_move_factor = 3.0
        config.fret_difficulty_base = 1.2
        config.fret_difficulty_curve = "exponential"
        config.multi_note_threshold = 2
        config.high_fret_penalty_factor = 2.0
        config.time_pressure_weight = 5.0
        return config


# ====================== 示例和测试 ======================

class ExampleSequences:
    """示例序列"""

    @staticmethod
    def get_examples():
        return {
            "single_note_scale": [
                [3, 25, 25, 25, 25, 25],
                [5, 25, 25, 25, 25, 25],
                [7, 25, 25, 25, 25, 25],
                [8, 25, 25, 25, 25, 25],
                [7, 25, 25, 25, 25, 25],
                [5, 25, 25, 25, 25, 25],
                [3, 25, 25, 25, 25, 25],
                [0, 25, 25, 25, 25, 25],
            ],

            "mixed_note_passage": [
                [3, 2, 0, 25, 25, 25],  # 3音符
                [5, 5, 5, 25, 25, 25],  # 3音符
                [7, 25, 25, 25, 25, 25],  # 单音符
                [8, 25, 25, 25, 25, 25],  # 单音符
                [7, 7, 7, 25, 25, 25],  # 3音符
                [5, 5, 25, 25, 25, 25],  # 2音符
                [3, 2, 0, 25, 25, 25],  # 3音符
                [25, 25, 25, 25, 25, 25],  # 无音符
            ],

            "complex_chord_progression": [
                [0, 1, 0, 2, 3, 0],  # C和弦 (4音符)
                [0, 2, 2, 0, 0, 0],  # Em和弦 (3音符)
                [2, 0, 0, 0, 3, 2],  # G和弦 (4音符)
                [0, 1, 0, 2, 3, 0],  # C和弦
                [3, 3, 4, 5, 5, 3],  # Am和弦 (6音符)
                [2, 3, 4, 0, 0, 0],  # D和弦 (4音符)
                [0, 1, 0, 2, 3, 0],  # C和弦
                [25, 25, 25, 25, 25, 25],  # 休息
            ],

            "high_fret_solo": [
                [15, 25, 25, 25, 25, 25],
                [17, 25, 25, 25, 25, 25],
                [19, 25, 25, 25, 25, 25],
                [20, 25, 25, 25, 25, 25],
                [19, 25, 25, 25, 25, 25],
                [17, 25, 25, 25, 25, 25],
                [15, 25, 25, 25, 25, 25],
                [12, 25, 25, 25, 25, 25],
            ],
        }


# ====================== 主程序 ======================

def main():
    """主演示函数"""
    print("=" * 70)
    print("吉他演奏难度分析系统 - 重构版")
    print("=" * 70)

    # 测试不同配置
    configs = {
        "默认配置": PresetConfigs.get_default(),
        "简单配置": PresetConfigs.get_easy(),
        "困难配置": PresetConfigs.get_hard(),
        "极限配置": PresetConfigs.get_extreme(),
    }

    sequences = ExampleSequences.get_examples()

    results = {}

    for config_name, config in configs.items():
        print(f"\n{'=' * 30} {config_name} {'=' * 30}")

        analyzer = GuitarSequenceAnalyzer(config)

        # 测试混合音符段落
        sequence = sequences["mixed_note_passage"]
        print(f"\n分析序列: 混合音符段落")

        try:
            report = analyzer.analyze_sequence(sequence)

            stats = report["statistics"]
            print(f"总体难度: {stats['overall_difficulty']:.1f}")
            print(f"难度等级: {stats['difficulty_level']}")
            print(f"平均位置难度: {stats['avg_position_difficulty']:.1f}")
            print(f"平均移动难度: {stats['avg_move_difficulty']:.1f}")
            print(f"多音符位置数: {report['multi_note_analysis']['total_multi_notes']}")

            # 显示移动因子统计
            if report["hardest_moves"]:
                hardest = report["hardest_moves"][0]
                print(f"最难移动: {hardest['from'] + 1}→{hardest['to'] + 1} "
                      f"(难度: {hardest['difficulty']:.1f}, 因子: {hardest['move_factor']:.1f})")

            results[config_name] = {
                "overall": stats['overall_difficulty'],
                "level": stats['difficulty_level'],
                "avg_move": stats['avg_move_difficulty']
            }

        except Exception as e:
            print(f"分析失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 配置比较
    print("\n" + "=" * 70)
    print("配置比较:")
    print("=" * 70)

    for config_name, result in results.items():
        print(f"{config_name:15} | 总体难度: {result['overall']:6.1f} | "
              f"等级: {result['level']:10} | 平均移动难度: {result['avg_move']:.1f}")

    # 演示自定义公式
    print("\n" + "=" * 70)
    print("自定义难度公式演示:")
    print("=" * 70)

    # 自定义公式：更加重视多音符移动
    def custom_formula(pos_diffs, move_diffs, accumulated, positions, moves):
        # 计算多音符移动的总难度
        multi_move_total = 0
        multi_move_count = 0

        for i, move in enumerate(moves):
            note_counts = move.get("note_counts", (0, 0))
            if note_counts[0] >= 2 or note_counts[1] >= 2:  # 至少一端是多音符
                multi_move_total += move_diffs[i]
                multi_move_count += 1

        # 基础计算
        base = np.mean(pos_diffs) * 0.3 + np.mean(move_diffs) * 0.7 if move_diffs else 0

        # 多音符移动加成
        if multi_move_count > 0:
            avg_multi_move = multi_move_total / multi_move_count
            base = base * 0.7 + avg_multi_move * 0.3

        return base * 1.5  # 整体放大

    # 使用自定义公式的配置
    custom_config = DifficultyConfig()
    custom_config.custom_difficulty_formula = custom_formula

    analyzer = GuitarSequenceAnalyzer(custom_config)
    sequence = sequences["complex_chord_progression"]
    report = analyzer.analyze_sequence(sequence)

    print(f"使用自定义公式的总体难度: {report['statistics']['overall_difficulty']:.1f}")
    print(f"难度等级: {report['statistics']['difficulty_level']}")
    print(f"多音符位置数: {report['multi_note_analysis']['total_multi_notes']}")

    # 显示建议
    print("\n练习建议:")
    for i, suggestion in enumerate(report["suggestions"][:5], 1):
        print(f"  {i}. {suggestion}")


# ====================== 工具函数 ======================

def analyze_with_config(sequence_data, config: DifficultyConfig) -> Dict:
    """使用指定配置分析序列"""
    analyzer = GuitarSequenceAnalyzer(config)
    return analyzer.analyze_sequence(sequence_data)


def compare_with_presets(sequence_data):
    """使用所有预置配置分析序列并比较"""
    configs = {
        "简单": PresetConfigs.get_easy(),
        "默认": PresetConfigs.get_default(),
        "困难": PresetConfigs.get_hard(),
        "极限": PresetConfigs.get_extreme(),
    }

    results = {}
    for name, config in configs.items():
        analyzer = GuitarSequenceAnalyzer(config)
        report = analyzer.analyze_sequence(sequence_data)
        results[name] = {
            "difficulty": report["statistics"]["overall_difficulty"],
            "level": report["statistics"]["difficulty_level"],
            "avg_move": report["statistics"]["avg_move_difficulty"]
        }

    return results


def create_custom_config(**kwargs) -> DifficultyConfig:
    """创建自定义配置"""
    config = DifficultyConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


if __name__ == "__main__":
    main()