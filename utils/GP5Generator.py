import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

import guitarpro as gp
from guitarpro.models import (
    NaturalHarmonic, ArtificialHarmonic, TappedHarmonic,
    PinchHarmonic, SemiHarmonic, SlideType, BendType,
    MeasureHeader, Duration, NoteType, BendEffect, BendPoint, TremoloPickingEffect,
    Beat, Note, Voice, Measure
)


class GuitarTechnique(Enum):
    """吉他演奏技巧枚举"""
    NORMAL = 0
    HAMMER_ON = 1
    PULL_OFF = 2
    SLIDE = 3
    BEND = 4
    VIBRATO = 5
    MUTE = 6
    NATURAL_HARMONIC = 7
    ARTIFICIAL_HARMONIC = 8
    TAPPED_HARMONIC = 9
    PINCH_HARMONIC = 10
    SEMI_HARMONIC = 11
    TREMOLO = 12
    PALM_MUTE = 13


class GuitarProGenerator:
    """
    GuitarPro文件生成器
    """

    def __init__(self):
        self.duration_map = self._create_duration_map()
        self.technique_mapper = self._create_technique_mapper()

    def _create_duration_map(self) -> Dict[str, Duration]:
        """创建时值映射 - 返回Duration对象"""
        return {
            'whole': Duration(value=Duration.whole),
            'half': Duration(value=Duration.half),
            'quarter': Duration(value=Duration.quarter),
            'eighth': Duration(value=Duration.eighth),
            'sixteenth': Duration(value=Duration.sixteenth),
            'thirty_second': Duration(value=Duration.thirtySecond),
            'sixty_fourth': Duration(value=Duration.sixtyFourth),
        }

    def _create_technique_mapper(self) -> Dict[GuitarTechnique, Any]:
        """创建技巧映射器"""
        return {
            GuitarTechnique.HAMMER_ON: self._create_hammer_on,
            GuitarTechnique.PULL_OFF: self._create_pull_off,
            GuitarTechnique.SLIDE: self._create_slide,
            GuitarTechnique.BEND: self._create_bend,
            GuitarTechnique.VIBRATO: self._create_vibrato,
            GuitarTechnique.MUTE: self._create_mute,
            GuitarTechnique.NATURAL_HARMONIC: self._create_natural_harmonic,
            GuitarTechnique.ARTIFICIAL_HARMONIC: self._create_artificial_harmonic,
            GuitarTechnique.TAPPED_HARMONIC: self._create_tapped_harmonic,
            GuitarTechnique.PINCH_HARMONIC: self._create_pinch_harmonic,
            GuitarTechnique.SEMI_HARMONIC: self._create_semi_harmonic,
            GuitarTechnique.TREMOLO: self._create_tremolo,
            GuitarTechnique.PALM_MUTE: self._create_palm_mute,
        }

    def create_empty_song(self, title="Generated Tab", artist="AI Composer",
                          tempo=120) -> gp.Song:
        FILE = Path(__file__).resolve()

        project_dir = os.path.dirname(FILE)

        song = gp.parse(os.path.join(project_dir, "example.gp5"))
        # reset measures
        for t in song.tracks:
            t.measures = []

        song.title = title
        song.artist = artist
        song.tempo = tempo

        return song

    def add_note(self, song: gp.Song, string: int, fret: int,
                 new_duration, technique: GuitarTechnique = GuitarTechnique.NORMAL,
                 position=-1,
                 track_index: int = 0, measure_index: int = -1, replace_existing=False, velocity = 127) -> bool:

        duration = None
        if isinstance(new_duration, str):
            duration = self.duration_map.get(new_duration)
            if duration is None:
                print(f"警告: 未知的时值 '{new_duration}'，使用四分音符")
                duration = self.duration_map['quarter']
        elif isinstance(new_duration, Duration):
            duration = new_duration
        assert isinstance(duration,Duration)

        track = song.tracks[track_index]

        if measure_index == -1:
            v = track.measures[-1].voices[0]
            if not v.isEmpty:
                last_beat = v.beats[-1]
                if self._get_beat_end(last_beat) == 4 * Duration.quarterTime:
                    self._add_measure(song, track)

        # 确保有足够的小节
        while len(track.measures) <= measure_index:
            self._add_measure(song, track)

        measure = track.measures[measure_index]

        # 确保有声部
        if not measure.voices:
            voice = Voice(measure)
            measure.voices.append(voice)

        voice = measure.voices[0]

        target_start = 0
        assert position < 4
        note = Note(None)
        note.velocity = velocity
        note.string = string
        note.value = fret
        note.type = NoteType.normal

        # check for existence
        if position >= 0:
            target_start = int(position * Duration.quarterTime)
            #print(f'target start = {target_start}')
            for idx, beat in enumerate(voice.beats):
                last = voice.beats[idx]
                last_duration = target_start - last.startInMeasure
                if last_duration == 0:  # if overlapped with existed beats
                    note.beat = beat
                    for n in beat.notes:
                        if n.string == note.string:
                            if not replace_existing:
                                note = n
                                # note = self.find_closest(note)
                            beat.notes.remove(n)
                            break
                    beat.notes.append(note)
                    break
                elif last_duration < 0:  # if exist beats after
                    beat_n = Beat(voice)
                    beat_n.duration = duration
                    beat_n.start = measure.start + target_start
                    assert beat_n.startInMeasure < 4 * Duration.quarterTime
                    note.beat = beat_n
                    beat_n.notes.append(note)
                    voice.beats.insert(idx, beat_n)
                    break
        if note.beat is None:
            # 创建拍子
            #print(f'failed to add note on position {position}, adding new note at {target_start}')
            beat = Beat(voice)

            beat.duration = duration  # 使用Duration对象
            if position == -1:
                target_start = voice.beats[-1].startInMeasure + voice.beats[-1].duration.time
            beat.start = measure.start + target_start
            # 创建音符
            note.beat = beat

            beat.notes.append(note)
            voice.beats.append(beat)

        # 应用技巧
        self.apply_technique(note, technique)

        #print(f'created note at {note.beat.startInMeasure}')
        return True

    def _add_measure(self, song: gp.Song, track: gp.Track):
        """添加新小节"""
        # 创建新的小节头
        last_header = song.measureHeaders[-1]
        header = MeasureHeader()
        header.number = last_header.number + 1
        header.timeSignature = last_header.timeSignature
        header.start = last_header.end

        song.measureHeaders.append(header)

        # 创建小节
        measure = Measure(track, header)
        voice = Voice(measure)
        measure.voices.append(voice)
        track.measures.append(measure)

    def _normalize(self, measure:Measure):
        voice = measure.voices[0]
        size = len(voice.beats)
        if size == 0 or voice.beats[0].startInMeasure != 0:  # if cavity before any beats
            beat_n = Beat(voice)
            beat_n.start = measure.start
            beat_n.duration = Duration.fromTime(Duration.quarterTime)
            voice.beats.insert(0, beat_n)
            size += 1

        for idx, beat in enumerate(voice.beats):
            assert beat.startInMeasure < Duration.quarterTime * 4
            duration = beat.duration
            next_b = voice.beats[idx + 1] if size > idx + 1 else None
            new_duration = (next_b.startInMeasure - beat.startInMeasure) if next_b is not None else \
                (Duration.quarterTime * 4 - beat.startInMeasure)  # take minimum possible duration
            if new_duration != duration.time:  # if not fit
                if new_duration < 0:
                    print("?")
                beat.duration = self.ticks_to_gp_duration(new_duration)
                #print(f'last beat at {idx} is update with duration {beat.duration.value} started from {beat.startInMeasure}')


    def _fill_empty(self, measure: Measure, end: int, extend: bool = True):
        """填充小节中的空白区域

        Args:
            measure: 要填充的小节
            end: 填充的结束位置（tick）
            extend: 是否从现有拍子扩展时值
        """
        voice = measure.voices[0]

        # 如果没有拍子，填充整个小节
        if not voice.beats:
            voice.beats = self._fit_empty_beat(end, 0, None)
            return

        size = len(voice.beats)

        for idx, beat in enumerate(voice.beats):
            beat_end = self._get_beat_end(beat)
            if idx + 1 < size:
                duration = voice.beats[idx + 1].startInMeasure - beat_end
                if duration > 0:
                    fill_beats = self._create_tie_beat(beat if extend else None, duration, beat_end + voice.measure.start)
                    for b in reversed(fill_beats):
                        b.voice = voice
                        voice.beats.insert(idx + 1, b)
                    print(f'填充空白: 从 {beat_end}, 时长 {duration}')
            else:
                last_beat_end = beat_end + voice.measure.start
                if last_beat_end < end:
                    start_fill = last_beat_end
                    duration_to_fill = end - start_fill



                    # 创建填充拍子
                    fill_beats = self._create_tie_beat(voice.beats[-1] if extend else None,duration_to_fill, start_fill,)

                    # 添加到小节末尾
                    voice.beats.extend(fill_beats)


    def _get_beat_end(self, beat: Beat) -> int:
        """获取拍子的结束位置（tick）"""
        if beat.start is None:
            return 0
        return beat.startInMeasure + beat.duration.time

    def _fit_empty_beat(self, duration: int, start: int, beat: Beat = None) -> List[Beat]:
        """将空白时长分割为合适的休止符拍子，使用最少数量的beat

        Args:
            duration: 要填充的总时长（tick）
            start: 开始位置（tick）
            beat: 参考拍子（用于扩展时值模式）

        Returns:
            休止符拍子列表（使用最少数量的beat）
        """
        d = duration
        beats = []
        current_start = start

        # 使用贪心算法，优先使用最大的时值来最小化beat数量
        # 包含附点音符的时值映射，按时长从大到小排序
        tick_mapping = [
            (Duration.quarterTime * 4, Duration.whole),  # 全音符
            (Duration.quarterTime * 3, Duration.half),  # 附点二分音符
            (Duration.quarterTime * 2, Duration.half),  # 二分音符
            (Duration.quarterTime * 1.5, Duration.quarter),  # 附点四分音符
            (Duration.quarterTime, Duration.quarter),  # 四分音符
            (Duration.quarterTime * 0.75, Duration.eighth),  # 附点八分音符
            (Duration.quarterTime // 2, Duration.eighth),  # 八分音符
            (Duration.quarterTime * 0.375, Duration.sixteenth),  # 附点十六分音符
            (Duration.quarterTime // 4, Duration.sixteenth),  # 十六分音符
            (Duration.quarterTime // 8, Duration.thirtySecond),  # 三十二分音符
            (Duration.quarterTime // 16, Duration.sixtyFourth),  # 六十四分音符
        ]

        # 如果提供了参考拍子，先尝试使用参考时值
        if beat is not None:
            ref_duration_time = beat.duration.time
            # 只使用参考时值如果它能完全填满剩余空间
            if d % ref_duration_time == 0:
                while d >= ref_duration_time:
                    rest_beat = self._create_rest_beat(beat.duration, current_start)
                    beats.append(rest_beat)
                    d -= ref_duration_time
                    current_start += ref_duration_time

        # 使用贪心算法选择最大的可用时值
        for tick_duration, gp_duration in tick_mapping:
            while d >= tick_duration:
                # 创建对应时值的 Duration 对象
                dur_obj = Duration(value=gp_duration)

                # 对于附点音符，需要设置isDotted属性
                if tick_duration in [Duration.quarterTime * 3,
                                     Duration.quarterTime * 1.5,
                                     Duration.quarterTime * 0.75,
                                     Duration.quarterTime * 0.375]:
                    dur_obj.isDotted = True

                rest_beat = self._create_rest_beat(dur_obj, current_start)
                beats.append(rest_beat)
                d -= tick_duration
                current_start += tick_duration

        # 如果还有剩余时长，使用最接近的时值填充
        if d > 0:
            # 找到最接近的时值
            closest_duration = None
            min_diff = float('inf')

            for tick_duration, gp_duration in tick_mapping:
                if tick_duration >= d:  # 只考虑比剩余时长大的时值
                    diff = tick_duration - d
                    if diff < min_diff:
                        min_diff = diff
                        closest_duration = (tick_duration, gp_duration)

            if closest_duration:
                tick_duration, gp_duration = closest_duration
                dur_obj = Duration(value=gp_duration)

                # 对于附点音符，需要设置isDotted属性
                if tick_duration in [Duration.quarterTime * 3,
                                     Duration.quarterTime * 1.5,
                                     Duration.quarterTime * 0.75,
                                     Duration.quarterTime * 0.375]:
                    dur_obj.isDotted = True

                rest_beat = self._create_rest_beat(dur_obj, current_start)
                beats.append(rest_beat)

        return beats

    def _create_tie_beat(self, previous_beat: Beat, duration: int, start: int) -> List[Beat]:
        """创建延音拍子，延续前一个拍子的音符

        Args:
            previous_beat: 前一个拍子（需要延长的拍子）
            duration: 延长的时长（tick）
            start: 开始位置（tick）

        Returns:
            延音拍子列表
        """
        try:
            # 如果前一个拍子没有音符，无法创建延音
            if not previous_beat.notes:
                return self._fit_empty_beat(duration, start, previous_beat)

            # 将延长时长分割为合适的时值
            duration_beats = self._fit_empty_beat(duration, start, previous_beat)

            # 为每个分割的拍子创建延音音符
            for beat in duration_beats:
                # 复制前一个拍子的所有音符到当前拍子
                for prev_note in previous_beat.notes:
                    # 创建延音音符
                    tie_note = Note(beat)
                    tie_note.string = prev_note.string
                    tie_note.value = prev_note.value
                    tie_note.type = NoteType.tie  # 设置为延音类型

                    # 复制其他属性
                    tie_note.velocity = prev_note.velocity
                    tie_note.effect = prev_note.effect  # 注意：可能需要深拷贝

                    beat.notes.append(tie_note)

            return duration_beats

        except Exception as e:
            print(f"创建延音拍子时出错: {e}")
            return []

    def _create_rest_beat(self, duration: Duration, start: int = None) -> Beat:
        """创建休止符拍子

        Args:
            duration: 时值
            start: 开始位置（可选）

        Returns:
            休止符拍子
        """
        # 创建虚拟的 Voice 用于初始化 Beat
        # 注意：这里需要传入一个 Voice 对象，但我们稍后会将其移除
        from guitarpro.models import Voice
        dummy_voice = Voice(None)  # 传入 None 作为临时方案

        beat = Beat(dummy_voice)
        beat.duration = duration
        beat.status = gp.BeatStatus.normal

        if start is not None:
            beat.start = start

        return beat

    def ticks_to_gp_duration(self, ticks: int) -> Duration:
        assert ticks > 0
        """将 tick 数转换为最接近的 Duration 对象

        Args:
            ticks: tick 数

        Returns:
            最接近的 Duration 对象
        """
        # 定义 tick 到 Duration 的映射
        tick_mapping = [
            (Duration.quarterTime * 4, Duration.whole, False),
            (Duration.quarterTime * 3, Duration.half, True),  # 附点二分音符
            (Duration.quarterTime * 2, Duration.half, False),
            (Duration.quarterTime * 1.5, Duration.quarter, True),  # 附点四分音符
            (Duration.quarterTime, Duration.quarter, False),
            (Duration.quarterTime * 0.75, Duration.eighth, True),  # 附点八分音符
            (Duration.quarterTime // 2, Duration.eighth, False),
            (Duration.quarterTime * 0.375, Duration.sixteenth, True),  # 附点十六分音符
            (Duration.quarterTime // 4, Duration.sixteenth, False),
            (Duration.quarterTime // 8, Duration.thirtySecond, False),
            (Duration.quarterTime // 16, Duration.sixtyFourth, False),
        ]

        # 找到最接近的标准时值
        best_duration = Duration.quarterTime  # 默认四分音符
        is_dotted = False
        min_diff = float('inf')

        for target_ticks, duration_value, dotted in tick_mapping:
            diff = ticks - target_ticks
            if 0 <= diff < min_diff:
                min_diff = diff
                best_duration = duration_value
                is_dotted = dotted
        assert min_diff != float('inf')
        #if min_diff != 0:
            #print(f'{min_diff} ticks from true value: getting {ticks} but expecting { 960 * 4 / best_duration}')

        return Duration(value=best_duration, isDotted=is_dotted)

    def fill_measure_to_end(self, measure: Measure):
        """填充小节到标准结束位置"""

        # 计算小节的应有长度
        time_signature = measure.header.timeSignature
        numerator = time_signature.numerator
        denominator_value = time_signature.denominator.value

        # 计算小节的标准长度（tick）
        if denominator_value == Duration.whole:
            measure_length = numerator * Duration.quarterTime * 4
        elif denominator_value == Duration.half:
            measure_length = numerator * Duration.quarterTime * 2
        elif denominator_value == Duration.quarter:
            measure_length = numerator * Duration.quarterTime
        elif denominator_value == Duration.eighth:
            measure_length = numerator * Duration.quarterTime // 2
        elif denominator_value == Duration.sixteenth:
            measure_length = numerator * Duration.quarterTime // 4
        else:
            measure_length = numerator * Duration.quarterTime  # 默认

        # 填充到标准长度
        self._fill_empty(measure, measure_length, extend=True)


    def post_process(self, song: gp.Song, track_index: int = 0):
        """填充所有小节的空白"""
        track = song.tracks[track_index]
        for measure in track.measures:
            self._normalize(measure)
            self.fill_measure_to_end(measure)

    # 谐波效果方法
    def _create_natural_harmonic(self, note: Note):
        """创建自然泛音"""
        note.effect.harmonic = NaturalHarmonic()

    def _create_artificial_harmonic(self, note: Note, pitch=None, octave=None):
        """创建人工泛音"""
        note.effect.harmonic = ArtificialHarmonic()

    def _create_tapped_harmonic(self, note: Note, fret=None):
        """创建点弦泛音"""
        harmonic = TappedHarmonic()
        if fret is not None:
            harmonic.fret = fret
        else:
            harmonic.fret = 12
        note.effect.harmonic = harmonic

    def _create_pinch_harmonic(self, note: Note):
        """创建拨片泛音"""
        note.effect.harmonic = PinchHarmonic()

    def _create_semi_harmonic(self, note: Note):
        """创建半泛音"""
        note.effect.harmonic = SemiHarmonic()

    # 其他效果方法
    def _create_hammer_on(self, note: Note):
        """创建击弦效果"""
        note.effect.hammer = True

    def _create_pull_off(self, note: Note):
        """创建勾弦效果"""
        note.effect.hammer = True

    def _create_slide(self, note: Note, slide_type=SlideType.legatoSlideTo):
        """创建滑音效果"""
        note.effect.slides.append(slide_type)

    def _create_bend(self, note: Note, bend_value=4):
        """创建弯音效果"""
        try:
            bend = BendEffect()
            bend.type = BendType.bend

            # 创建弯音点
            point1 = BendPoint()
            point1.position = 0
            point1.value = 0

            point2 = BendPoint()
            point2.position = 60
            point2.value = bend_value * 25

            bend.points = [point1, point2]
            note.effect.bend = bend
        except Exception as e:
            print(f"创建弯音效果时出错: {e}")

    def _create_vibrato(self, note: Note):
        """创建颤音效果"""
        note.effect.vibrato = True

    def _create_mute(self, note: Note):
        """创建闷音效果"""
        note.effect.ghostNote = True
        note.type = NoteType.dead

    def _create_tremolo(self, note: Note, duration_name='sixteenth'):
        """创建颤音拨弦效果"""
        try:
            duration = self.duration_map.get(duration_name, self.duration_map['sixteenth'])
            note.effect.tremoloPicking = TremoloPickingEffect()
            note.effect.tremoloPicking.duration = duration
        except Exception as e:
            print(f"创建颤音拨弦效果时出错: {e}")

    def _create_palm_mute(self, note: Note):
        """创建手掌闷音效果"""
        note.effect.palmMute = True

    def apply_technique(self, note: Note, technique: GuitarTechnique, **kwargs):
        """应用演奏技巧到音符"""
        try:
            if technique in self.technique_mapper:
                self.technique_mapper[technique](note, **kwargs)
        except Exception as e:
            print(f"应用技巧 {technique} 时出错: {e}")

    def generate_simple_tab(self, output_path: str = "simple_tab.gp5"):
        """生成简单的测试吉他谱"""
        # try:
        song = self.create_empty_song("Test Guitar Tab", "GP Generator")

        # 添加一些演示音符 - 使用正确的时值名称
        notes = [
            (1, 0, 'quarter', 0, GuitarTechnique.NORMAL),
            #(2, 1, 'eighth', 0.25, GuitarTechnique.HAMMER_ON),
            #(2, 3, 'eighth', 0.5, GuitarTechnique.PULL_OFF),
            #(3, 2, 'quarter', 0.75, GuitarTechnique.BEND),
            #(4, 0, 'half', 0, GuitarTechnique.PALM_MUTE),
            (1, 2, 'quarter', 3.25, GuitarTechnique.ARTIFICIAL_HARMONIC),
        ]

        print("添加音符:")
        for i, (string, fret, duration, position, technique) in enumerate(notes):
            success = self.add_note(song, string, fret, duration, technique, position, measure_index=0)
            if success:
                print(f"  ✓ {string}弦{fret}品 - {duration} - {technique.name}")
            else:
                print(f"  ✗ 添加音符失败: {string}弦{fret}品")

        self.post_process(song)

        # 保存文件
        gp.write(song, output_path)
        print(f"✓ 文件已保存: {output_path}")
        return output_path

    def generate_scale_tab(self, output_path: str = "scale.gp5"):
        """生成音阶练习谱"""
        # try:
        song = self.create_empty_song()

        # self.create_empty_song("Major Scale Exercise", "GP Generator")

        # C大调音阶在5弦上的指法
        c_major_scale = [
            (3, 0, 'quarter', -1, GuitarTechnique.NORMAL),  # G
            (3, 2, 'quarter', 2, GuitarTechnique.NORMAL),  # A
            (4, 4, 'quarter', 0, GuitarTechnique.NORMAL),  # B
            (2, 1, 'quarter', 2.5, GuitarTechnique.NORMAL),  # C
            (2, 3, 'quarter', 1.25, GuitarTechnique.NORMAL),  # D
            (2, 5, 'quarter', 1, GuitarTechnique.NORMAL),  # E
            (1, 0, 'quarter', -1, GuitarTechnique.NORMAL),  # F
            (1, 2, 'sixteenth', 0.25, GuitarTechnique.NORMAL),  # G
        ]

        print("生成C大调音阶:")
        for i, (string, fret, duration, position, technique) in enumerate(c_major_scale):
            measure_index = i // 4  # 每4个音符一个小节
            self.add_note(song, string, fret, duration, technique, position, measure_index=measure_index)
            print(f"  {string}弦{fret}品", end=" → " if i < len(c_major_scale) - 1 else "\n")

        gp.write(song, output_path)
        print(f"✓ 音阶文件已保存: {output_path}")
        return output_path

        '''except Exception as e:
            print(f"生成音阶时出错: {e}")
            return None'''


# 使用示例
if __name__ == "__main__":
    example = gp.parse("Daisy.gp5")

    # 创建生成器
    generator = GuitarProGenerator()

    print("=== GuitarPro 生成器测试 ===")

    # 生成简单测试谱
    output_file = generator.generate_simple_tab("test_output.gp5")

    # 生成音阶练习谱
    scale_file = generator.generate_scale_tab("c_major_scale.gp5")

    print("\n=== 可用的时值 ===")
    for duration_name in generator.duration_map.keys():
        print(f"- {duration_name}")

    print("\n=== 可用的技巧 ===")
    for technique in GuitarTechnique:
        print(f"- {technique.name}")

    if output_file:
        print(f"\n✓ 测试完成！生成的文件: {output_file}")
    else:
        print(f"\n✗ 生成失败")
