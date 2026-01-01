import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json


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
    """位置分析器"""

    def __init__(self, config: DifficultyConfig, hand_model: HandModel):
        self.config = config
        self.hand_model = hand_model

    def analyze_position(self, frets: List[int]) -> Dict[str, Any]:
        """分析单个位置"""
        # 提取有效音符
        active_frets = [(i, f) for i, f in enumerate(frets) if 0 <= f < self.config.non_playing_fret]
        active_strings = [s for s, _ in active_frets]
        active_fret_values = [f for _, f in active_frets]
        note_count = len(active_frets)

        # 基础信息
        position_info = {
            "frets": frets,
            "active_frets": active_frets,
            "active_fret_values": active_fret_values,
            "active_strings": active_strings,
            "note_count": note_count,
            "is_single_note": note_count == 1,
            "is_multi_note": note_count >= self.config.multi_note_threshold
        }

        if note_count == 0:
            position_info.update({
                "base_difficulty": 0.0,
                "fret_difficulty": 0.0,
                "stretch_difficulty": 0.0,
                "note_count_difficulty": 0.0,
                "total_difficulty": 0.0
            })
            return position_info

        # 计算各项难度
        base_difficulty = self.config.position_base

        # 品位难度（取消底品难度）
        fret_difficulty = 0.0
        for fret in active_fret_values:
            fret_difficulty += self.hand_model.calculate_fret_difficulty(fret)
        if active_fret_values:
            fret_difficulty /= len(active_fret_values)  # 平均品位难度

        # 伸展难度
        stretch_difficulty = self.hand_model.calculate_stretch_difficulty(active_fret_values)

        # 音符数量难度
        note_count_difficulty = self.hand_model.calculate_note_count_difficulty(note_count)

        # 总位置难度
        total_difficulty = (
                base_difficulty +
                fret_difficulty +
                stretch_difficulty +
                note_count_difficulty
        )

        position_info.update({
            "base_difficulty": base_difficulty,
            "fret_difficulty": fret_difficulty,
            "stretch_difficulty": stretch_difficulty,
            "note_count_difficulty": note_count_difficulty,
            "total_difficulty": total_difficulty
        })

        return position_info


# ====================== 移动分析 ======================

class MoveAnalyzer:
    """移动分析器"""

    def __init__(self, config: DifficultyConfig):
        self.config = config

    def calculate_center(self, active_frets: List[Tuple[int, int]]) -> Tuple[float, float]:
        """计算手位中心（弦，品）"""
        if not active_frets:
            return (0.0, 0.0)

        weighted_string = 0.0
        weighted_fret = 0.0
        total_weight = 0.0

        for string, fret in active_frets:
            # 权重：低音弦权重更高，高品位权重更高
            string_weight = (6 - string) * 0.3 + 0.7
            fret_weight = 1.0 + (fret / 24) * 1.0
            weight = string_weight * fret_weight

            weighted_string += string * weight
            weighted_fret += fret * weight
            total_weight += weight

        return (
            weighted_string / total_weight if total_weight > 0 else 0.0,
            weighted_fret / total_weight if total_weight > 0 else 0.0
        )

    def calculate_move_distance(self, pos1_center: Tuple[float, float],
                                pos2_center: Tuple[float, float]) -> float:
        """计算移动距离"""
        string_distance = abs(pos2_center[0] - pos1_center[0])
        fret_distance = abs(pos2_center[1] - pos1_center[1])

        # 综合距离（弦和品都重要）
        distance = math.sqrt(string_distance ** 2 + fret_distance ** 2 * 4)
        return distance

    def calculate_move_difficulty(self, pos1_info: Dict, pos2_info: Dict,
                                  time_interval: float) -> Dict[str, Any]:
        """计算移动难度"""

        # 基础移动难度
        base_move_difficulty = self.config.move_base_difficulty

        # 音符数量因子
        note_count1 = pos1_info["note_count"]
        note_count2 = pos2_info["note_count"]

        if note_count1 == 0 and note_count2 == 0:
            move_factor = 0.0
        elif note_count1 == 1 and note_count2 == 1:
            move_factor = self.config.single_note_move_factor
        elif note_count1 >= self.config.multi_note_threshold and note_count2 >= self.config.multi_note_threshold:
            move_factor = self.config.multi_note_move_factor
        else:
            move_factor = self.config.mixed_note_move_factor

        # 距离难度
        distance_difficulty = 0.0
        if pos1_info["active_frets"] and pos2_info["active_frets"]:
            center1 = self.calculate_center(pos1_info["active_frets"])
            center2 = self.calculate_center(pos2_info["active_frets"])
            distance = self.calculate_move_distance(center1, center2)
            distance_difficulty = distance * self.config.move_distance_weight * move_factor

        # 模式变化难度
        pattern_difficulty = self.calculate_pattern_difficulty(pos1_info, pos2_info) * move_factor

        # 时间压力
        time_pressure = 0.0
        if self.config.transition_time_pressure:
            sixteenth_duration = 60 / self.config.bpm / 4
            if time_interval < sixteenth_duration * self.config.time_pressure_threshold:
                pressure = (
                                       sixteenth_duration * self.config.time_pressure_threshold - time_interval) / sixteenth_duration
                time_pressure = pressure * self.config.time_pressure_weight

        # 总移动难度
        total_move_difficulty = (
                base_move_difficulty +
                distance_difficulty +
                pattern_difficulty +
                time_pressure
        )

        return {
            "total_difficulty": total_move_difficulty,
            "components": {
                "base_move": base_move_difficulty,
                "distance": distance_difficulty,
                "pattern": pattern_difficulty,
                "time_pressure": time_pressure
            },
            "move_factor": move_factor,
            "note_counts": (note_count1, note_count2)
        }

    def calculate_pattern_difficulty(self, pos1_info: Dict, pos2_info: Dict) -> float:
        """计算模式变化难度"""
        strings1 = set(pos1_info["active_strings"])
        strings2 = set(pos2_info["active_strings"])

        if not strings1 or not strings2:
            return 0.5  # 中等变化

        # 共同弦比例
        common_strings = strings1.intersection(strings2)
        total_strings = strings1.union(strings2)

        if not total_strings:
            return 0.0

        # 共同弦越少，模式变化越大
        similarity = len(common_strings) / len(total_strings)
        pattern_change = (1 - similarity) * self.config.move_pattern_weight

        # 音符数量变化
        count_change = abs(pos1_info["note_count"] - pos2_info["note_count"])
        pattern_change += count_change * 0.2

        return pattern_change


# ====================== 序列分析 ======================

class GuitarSequenceAnalyzer:
    """序列分析器"""

    def __init__(self, config: DifficultyConfig):
        self.config = config
        self.hand_model = HandModel(config)
        self.position_analyzer = PositionAnalyzer(config, self.hand_model)
        self.move_analyzer = MoveAnalyzer(config)

        # 状态
        self.stamina = 100.0
        self.accumulated_difficulty = 0.0

    def parse_sequence(self, sequence_data) -> List[Dict]:
        """解析序列数据"""
        parsed = []

        for i, chord_data in enumerate(sequence_data):
            if isinstance(chord_data, (list, tuple)) and len(chord_data) == 6:
                frets = list(chord_data)

                # 时间位置
                sixteenth_duration = 60 / self.config.bpm / 4
                time_position = i * sixteenth_duration
                measure_num = i // self.config.measure_length
                measure_pos = i % self.config.measure_length

                parsed.append({
                    "index": i,
                    "time": time_position,
                    "frets": frets,
                    "measure_num": measure_num,
                    "measure_pos": measure_pos,
                    "is_line_break": measure_pos == self.config.measure_length - 1 and i < len(sequence_data) - 1
                })
            else:
                raise ValueError(f"第{i + 1}个和弦格式错误: {chord_data}")

        return parsed

    def analyze_sequence(self, sequence_data) -> Dict[str, Any]:
        """分析整个序列"""
        # 重置状态
        self.stamina = 100.0
        self.accumulated_difficulty = 0.0

        # 解析序列
        sequence = self.parse_sequence(sequence_data)

        # 分析每个位置
        positions = []
        position_difficulties = []

        for chord_info in sequence:
            # 位置分析
            pos_info = self.position_analyzer.analyze_position(chord_info["frets"])
            pos_info.update({
                "time": chord_info["time"],
                "index": chord_info["index"],
                "measure_num": chord_info["measure_num"],
                "is_line_break": chord_info["is_line_break"]
            })

            positions.append(pos_info)
            position_difficulties.append(pos_info["total_difficulty"])

            # 耐力消耗
            self.stamina -= pos_info["total_difficulty"] * self.config.stamina_drain_factor
            self.stamina = max(0.0, self.stamina)

        # 分析移动
        moves = []
        move_difficulties = []

        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]

            time_interval = pos2["time"] - pos1["time"]
            if time_interval <= 0:
                time_interval = 60 / self.config.bpm / 4

            # 移动分析
            move_info = self.move_analyzer.calculate_move_difficulty(pos1, pos2, time_interval)
            move_info.update({
                "from_index": i,
                "to_index": i + 1,
                "is_cross_measure": pos1["measure_num"] != pos2["measure_num"]
            })

            moves.append(move_info)
            move_difficulties.append(move_info["total_difficulty"])

        # 计算累计难度
        accumulated = self.calculate_accumulated_difficulty(position_difficulties, move_difficulties)

        # 计算总体难度
        overall_difficulty = self.calculate_overall_difficulty(
            position_difficulties, move_difficulties, accumulated, positions, moves
        )

        # 确定难度等级
        difficulty_level = DifficultyLevel.from_score(overall_difficulty, self.config.difficulty_thresholds)

        # 生成报告
        report = self.generate_report(
            positions, moves, overall_difficulty, difficulty_level,
            position_difficulties, move_difficulties, accumulated
        )

        return report

    def calculate_accumulated_difficulty(self, pos_diffs, move_diffs):
        """计算累计难度"""
        if not self.config.accumulated_difficulty_enabled:
            return [0.0] * len(pos_diffs)

        accumulated = []
        current = 0.0

        for i in range(len(pos_diffs)):
            current += pos_diffs[i]
            if i > 0:
                current += move_diffs[i - 1]

            # 轻微衰减
            if i % 4 == 0:
                current *= 0.95

            accumulated.append(current)

        return accumulated

    def calculate_overall_difficulty(self, pos_diffs, move_diffs, accumulated, positions, moves):
        """计算总体难度"""
        if self.config.custom_difficulty_formula:
            return self.config.custom_difficulty_formula(
                pos_diffs, move_diffs, accumulated, positions, moves
            )

        # 默认计算公式
        if not pos_diffs:
            return 0.0

        # 位置难度统计
        avg_pos = np.mean(pos_diffs) if pos_diffs else 0.0
        max_pos = max(pos_diffs) if pos_diffs else 0.0

        # 移动难度统计
        avg_move = np.mean(move_diffs) if move_diffs else 0.0
        max_move = max(move_diffs) if move_diffs else 0.0

        # 累计难度峰值
        max_accumulated = max(accumulated) if accumulated else 0.0

        # 多音符难度加成
        multi_note_count = sum(1 for p in positions if p["note_count"] >= self.config.multi_note_threshold)
        multi_note_factor = 1.0 + (multi_note_count / len(positions)) * 0.5

        # 跨小节难度加成
        cross_measure_count = sum(1 for m in moves if m.get("is_cross_measure", False))
        cross_measure_factor = 1.0 + (cross_measure_count / max(len(moves), 1)) * 0.3

        # 综合计算
        overall = (
                          avg_pos * 0.25 +
                          max_pos * 0.15 +
                          avg_move * 0.30 +
                          max_move * 0.20 +
                          max_accumulated * self.config.accumulated_weight
                  ) * multi_note_factor * cross_measure_factor

        return overall

    def generate_report(self, positions, moves, overall_diff, difficulty_level,
                        pos_diffs, move_diffs, accumulated):
        """生成详细报告"""

        # 统计信息
        stats = {
            "overall_difficulty": overall_diff,
            "difficulty_level": difficulty_level.value,
            "bpm": self.config.bpm,
            "total_positions": len(positions),
            "total_moves": len(moves),
            "final_stamina": self.stamina,
            "avg_position_difficulty": np.mean(pos_diffs) if pos_diffs else 0.0,
            "max_position_difficulty": max(pos_diffs) if pos_diffs else 0.0,
            "avg_move_difficulty": np.mean(move_diffs) if move_diffs else 0.0,
            "max_move_difficulty": max(move_diffs) if move_diffs else 0.0,
            "max_accumulated_difficulty": max(accumulated) if accumulated else 0.0
        }

        # 最难位置
        hardest_positions = []
        if positions:
            sorted_indices = np.argsort(pos_diffs)[-3:][::-1]
            hardest_positions = [
                {
                    "index": positions[i]["index"],
                    "difficulty": pos_diffs[i],
                    "note_count": positions[i]["note_count"]
                }
                for i in sorted_indices
            ]

        # 最难移动
        hardest_moves = []
        if moves:
            move_diffs_arr = np.array([m["total_difficulty"] for m in moves])
            sorted_indices = np.argsort(move_diffs_arr)[-3:][::-1]
            hardest_moves = [
                {
                    "from": moves[i]["from_index"],
                    "to": moves[i]["to_index"],
                    "difficulty": move_diffs_arr[i],
                    "move_factor": moves[i].get("move_factor", 1.0),
                    "note_counts": moves[i].get("note_counts", (0, 0))
                }
                for i in sorted_indices
            ]

        # 多音符统计
        multi_note_stats = {
            "total_multi_notes": sum(1 for p in positions if p["note_count"] >= self.config.multi_note_threshold),
            "max_notes_in_position": max(p["note_count"] for p in positions) if positions else 0
        }

        # 建议
        suggestions = self.generate_suggestions(overall_diff, difficulty_level,
                                                hardest_positions, hardest_moves,
                                                multi_note_stats)

        return {
            "statistics": stats,
            "hardest_positions": hardest_positions,
            "hardest_moves": hardest_moves,
            "multi_note_analysis": multi_note_stats,
            "suggestions": suggestions,
            "config_summary": self.config.to_dict()
        }

    def generate_suggestions(self, overall_diff, difficulty_level, hardest_positions,
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