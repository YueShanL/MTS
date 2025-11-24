import guitarpro as gp
from guitarpro.models import (
    NaturalHarmonic, ArtificialHarmonic, TappedHarmonic,
    PinchHarmonic, SemiHarmonic, SlideType, BendType,
    MidiChannel, GuitarString, MeasureHeader, TimeSignature,
    Duration, NoteType, BendEffect, BendPoint, TremoloPickingEffect,
    Beat, Note, Voice, Measure
)
from typing import List, Dict, Any, Optional
from enum import Enum


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
    完全修正的GuitarPro文件生成器
    解决Duration对象使用问题
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

        song = gp.parse("example.gp5")
        # reset measures
        for t in song.tracks:
            t.measures = []

        song.title = title
        song.artist = artist
        song.tempo = tempo


        '''"""创建空的歌曲结构"""
        # 创建歌曲
        song = gp.Song()
        song.title = title
        song.artist = artist
        song.tempo = tempo

        # 创建MIDI通道
        channel = MidiChannel()
        channel.channel = 0
        channel.effectChannel = 1
        channel.instrument = 25  # 钢弦吉他
        channel.volume = 104
        channel.balance = 64

        # 创建吉他音轨
        track = gp.Track(song)
        track.name = "Guitar"
        track.channel = channel
        track.port = 0

        # 设置吉他弦 (标准调音 EADGBE)
        track.strings = [
            GuitarString(1, 64),  # 高E弦
            GuitarString(2, 59),  # B弦
            GuitarString(3, 55),  # G弦
            GuitarString(4, 50),  # D弦
            GuitarString(5, 45),  # A弦
            GuitarString(6, 40),  # 低E弦
        ]

        # 创建拍号
        time_signature = TimeSignature()
        time_signature.numerator = 4
        time_signature.denominator = Duration(value=Duration.quarter)

        # 创建小节头
        header = MeasureHeader()
        header.timeSignature = time_signature
        header.number = 1
        header.start = Duration.quarterTime

        song.measureHeaders.append(header)

        # 创建小节
        measure = Measure(track, header)
        voice = Voice(measure)
        measure.voices.append(voice)
        track.measures.append(measure)

        song.tracks.append(track)'''

        return song


    def add_note(self, song: gp.Song, string: int, fret: int,
                 duration_name: str, technique: GuitarTechnique = GuitarTechnique.NORMAL,
                 track_index: int = 0, measure_index: int = 0) -> bool:
        """添加音符到指定位置 - 修正版"""
        try:
            # 获取正确的Duration对象
            duration = self.duration_map.get(duration_name)
            if duration is None:
                print(f"警告: 未知的时值 '{duration_name}'，使用四分音符")
                duration = self.duration_map['quarter']

            track = song.tracks[track_index]

            # 确保有足够的小节
            while len(track.measures) <= measure_index:
                self._add_measure(song, track)

            measure = track.measures[measure_index]

            # 确保有声部
            if not measure.voices:
                voice = Voice(measure)
                measure.voices.append(voice)

            voice = measure.voices[0]

            # 创建拍子
            beat = Beat(voice)
            beat.duration = duration  # 使用Duration对象，不是整数

            # 创建音符
            note = Note(beat)
            note.string = string
            note.value = fret
            note.type = NoteType.normal

            beat.notes.append(note)
            voice.beats.append(beat)

            # 应用技巧
            self.apply_technique(note, technique)

            return True

        except Exception as e:
            print(f"添加音符时出错: {e}")
            return False

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
        try:
            song = self.create_empty_song("Test Guitar Tab", "GP Generator")

            # 添加一些演示音符 - 使用正确的时值名称
            notes = [
                (1, 0, 'quarter', GuitarTechnique.NORMAL),
                (2, 1, 'eighth', GuitarTechnique.HAMMER_ON),
                (2, 3, 'eighth', GuitarTechnique.PULL_OFF),
                (3, 2, 'quarter', GuitarTechnique.BEND),
                (4, 0, 'half', GuitarTechnique.PALM_MUTE),
                (1, 12, 'quarter', GuitarTechnique.NATURAL_HARMONIC),
            ]

            print("添加音符:")
            for i, (string, fret, duration, technique) in enumerate(notes):
                success = self.add_note(song, string, fret, duration, technique, measure_index=0)
                if success:
                    print(f"  ✓ {string}弦{fret}品 - {duration} - {technique.name}")
                else:
                    print(f"  ✗ 添加音符失败: {string}弦{fret}品")

            # 保存文件
            gp.write(song, output_path)
            print(f"✓ 文件已保存: {output_path}")
            return output_path

        except Exception as e:
            print(f"生成吉他谱时出错: {e}")
            return None

    def generate_scale_tab(self, output_path: str = "scale.gp5"):
        """生成音阶练习谱"""
        try:
            song = self.create_empty_song()

            # self.create_empty_song("Major Scale Exercise", "GP Generator")

            # C大调音阶在5弦上的指法
            c_major_scale = [
                (3, 0, 'quarter', GuitarTechnique.NORMAL),  # G
                (3, 2, 'quarter', GuitarTechnique.NORMAL),  # A
                (3, 4, 'quarter', GuitarTechnique.NORMAL),  # B
                (2, 1, 'quarter', GuitarTechnique.NORMAL),  # C
                (2, 3, 'quarter', GuitarTechnique.NORMAL),  # D
                (2, 5, 'quarter', GuitarTechnique.NORMAL),  # E
                (1, 0, 'quarter', GuitarTechnique.NORMAL),  # F
                (1, 2, 'quarter', GuitarTechnique.NORMAL),  # G
            ]

            print("生成C大调音阶:")
            for i, (string, fret, duration, technique) in enumerate(c_major_scale):
                measure_index = i // 4  # 每4个音符一个小节
                self.add_note(song, string, fret, duration, technique, measure_index=measure_index)
                print(f"  {string}弦{fret}品", end=" → " if i < len(c_major_scale) - 1 else "\n")

            gp.write(song, output_path)
            print(f"✓ 音阶文件已保存: {output_path}")
            return output_path

        except Exception as e:
            print(f"生成音阶时出错: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    example = gp.parse("example.gp5")

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
