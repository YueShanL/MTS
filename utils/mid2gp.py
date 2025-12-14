import mido
import guitarpro as gp
from guitarpro import Song
from guitarpro.gp5 import GP5File
from guitarpro.models import *
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import os

from fractions import Fraction

from utils.GP5Generator import GuitarProGenerator


class MIDItoGP5Converter:
    duration_map = [
        (4.0, 'whole', False),
        (3.0, 'half', True),  # 附点二分音符
        (2.0, 'half', False),
        (1.5, 'quarter', True),  # 附点四分音符
        (1.0, 'quarter', False),
        (0.75, 'eighth', True),  # 附点八分音符
        (0.5, 'eighth', False),
        (0.375, 'sixteenth', True),  # 附点十六分音符
        (0.25, 'sixteenth', False),
        (0.1875, 'thirtySecond', True),  # 附点三十二分音符
        (0.125, 'thirtySecond', False),
        (0.0625, 'sixtyFourth', False),
    ]

    def __init__(self, gp_generator: 'GuitarProGenerator', tuning: List[int] = None):
        self.gp_generator = gp_generator
        self.tuning = tuning or [64, 59, 55, 50, 45, 40]
        self.string_count = len(self.tuning)
        self.note_mappings = self._create_note_mappings()
        self.chord_detector = ChordDetector()

    def _create_note_mappings(self) -> Dict[int, List[Tuple[int, int]]]:
        mappings = {}

        for midi_note in range(40, 88):
            positions = []
            for string_idx, open_note in enumerate(self.tuning):
                fret = midi_note - open_note
                if 0 <= fret <= 24:
                    positions.append((string_idx + 1, fret))

            if positions:
                mappings[midi_note] = positions

        return mappings

    def parse_midi_file(self, midi_path: str, chord_tolerance: int = 10, target_tempo: int = 120, quantize_unit= 1/4) -> Dict:
        midi_data = mido.MidiFile(midi_path)

        tracks_data = []
        ticks_per_beat = midi_data.ticks_per_beat
        tempo = target_tempo
        ratio = 1.0

        track_notes = []
        current_time = 0
        active_notes = {}

        for msg in midi_data.merged_track:
            current_time += msg.time * ratio

            if msg.type == 'note_on' and msg.velocity > 0:
                # tempo = msg.tempo
                active_notes[msg.note] = {
                    'start_time': current_time,
                    'velocity': msg.velocity,
                    'channel': msg.channel,
                }

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    note_info = active_notes[msg.note]
                    duration_ticks = current_time - note_info['start_time']

                    start_beat = note_info['start_time'] / ticks_per_beat
                    duration_beat = duration_ticks / ticks_per_beat

                    if duration_ticks > 0:
                        track_notes.append({
                            'note': msg.note,
                            'start_time': start_beat,
                            'duration_ticks': duration_beat,
                            'velocity': note_info['velocity'],
                            'channel': note_info['channel']
                        })

                    del active_notes[msg.note]
            elif msg.type == 'set_tempo':
                # 记录变速事件
                ratio = tempo / (60000000 / msg.tempo)

        if track_notes:
            track_notes = self._quantize(track_notes, unit=quantize_unit)
            # 检测和弦
            chords = self._detect_chords_in_track(track_notes, chord_tolerance)

            tracks_data.append({
                'name': f'merged track',
                'notes': sorted(track_notes, key=lambda x: x['start_time']),
                'chords': chords
            })


        return {
            'ticks_per_beat': midi_data.ticks_per_beat,
            'tracks': tracks_data
        }

    def _quantize(self, notes: List[Dict], unit=duration_map[8][0]):

        for note in notes:  # round to nearest 1/16 by default
            note['duration_ticks'] = round(note['duration_ticks'] / unit, 0) * unit
            note['start_time'] = round(note['start_time'] / unit, 0) * unit

        return notes

    def _detect_chords_in_track(self, notes: List[Dict], tolerance: int) -> List[Dict]:
        if not notes:
            return []

        chords = []
        current_chord_notes = []
        current_time_window = None

        sorted_notes = sorted(notes, key=lambda x: x['start_time'])

        for note in sorted_notes:
            if current_time_window is None:
                current_time_window = (note['start_time'], note['start_time'] + tolerance)
                current_chord_notes = [note]
            elif note['start_time'] <= current_time_window[1]:
                current_chord_notes.append(note)
                current_time_window = (current_time_window[0], note['start_time'] + tolerance)
            else:
                if len(current_chord_notes) >= 2:
                    chord_info = self._analyze_chord(current_chord_notes)
                    chords.append(chord_info)

                current_time_window = (note['start_time'], note['start_time'] + tolerance)
                current_chord_notes = [note]

        if current_chord_notes and len(current_chord_notes) >= 2:
            chord_info = self._analyze_chord(current_chord_notes)
            chords.append(chord_info)

        return chords

    def _analyze_chord(self, chord_notes: List[Dict]) -> Dict:
        note_values = [note['note'] for note in chord_notes]
        start_time = chord_notes[0]['start_time']

        chord_name = self.chord_detector.detect_chord(note_values)
        chord_positions = self._find_chord_positions(note_values)

        return {
            'start_time': start_time,
            'notes': note_values,
            'name': chord_name,
            'positions': chord_positions,
            'duration_ticks': min(note['duration_ticks'] for note in chord_notes)
        }

    def _find_chord_positions(self, midi_notes: List[int]) -> List[Tuple[int, int]]:
        if not midi_notes:
            return []

        positions = []
        used_strings = set()

        sorted_notes = sorted(midi_notes)

        for note in sorted_notes:
            if note not in self.note_mappings:
                continue

            possible_positions = self.note_mappings[note]

            available_positions = [pos for pos in possible_positions if pos[0] not in used_strings]

            if available_positions:
                best_pos = min(available_positions, key=lambda x: x[1])
                positions.append(best_pos)
                used_strings.add(best_pos[0])
            else:
                best_pos = min(possible_positions, key=lambda x: (x[1], x[0]))
                positions.append(best_pos)
                used_strings.add(best_pos[0])

        return positions

    def find_best_position(self, midi_note: int, previous_positions: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        if midi_note not in self.note_mappings:
            closest_note = min(self.note_mappings.keys(),
                               key=lambda x: abs(x - midi_note))
            midi_note = closest_note

        possible_positions = self.note_mappings[midi_note]

        if not previous_positions:
            return min(possible_positions, key=lambda x: x[1])

        best_position = min(possible_positions,
                            key=lambda pos: self._calculate_position_cost(pos, previous_positions))

        return best_position

    def _calculate_position_cost(self, position: Tuple[int, int],
                                 previous_positions: List[Tuple[int, int]]) -> float:
        if not previous_positions:
            return 0

        min_cost = float('inf')
        for prev_pos in previous_positions[-3:]:
            string_diff = abs(position[0] - prev_pos[0])
            fret_diff = abs(position[1] - prev_pos[1])
            cost = string_diff * 2 + fret_diff
            min_cost = min(min_cost, cost)

        return min_cost

    def convert_midi_to_gp5(self, midi_path: str, output_path: str = None,
                            post_process = True,
                            title: str = None, artist: str = "MIDI Converter",
                            chord_tolerance: int = 10,
                            time_signature: Tuple[int, int] = (4, 4)) -> Song:
        """
        修复节拍问题的转换方法
        """
        print(f"开始转换: {midi_path}")

        midi_data = self.parse_midi_file(midi_path, chord_tolerance)
        ticks_per_beat = midi_data['ticks_per_beat']

        if title is None:
            title = os.path.basename(midi_path).replace('.mid', '')

        # 创建歌曲时指定拍号
        song = self.gp_generator.create_empty_song(title=title, artist=artist)

        # 设置拍号
        self._set_time_signature(song, time_signature)

        # 处理每个轨道
        for track_idx, track_data in enumerate(midi_data['tracks']):
            print(f"处理轨道 {track_idx + 1}: {track_data['name']}")
            print(f"  检测到 {len(track_data['chords'])} 个和弦")

            self._process_track(song, track_data, ticks_per_beat, track_idx, time_signature)

        if post_process:
            self.gp_generator.post_process(song)

        if output_path is not None:
            gp.write(song, output_path)
            print(f"转换完成: {output_path}")

        return song

    def _set_time_signature(self, song: gp.Song, time_signature: Tuple[int, int]):
        """设置拍号"""
        numerator, denominator = time_signature

        for header in song.measureHeaders:
            header.timeSignature.numerator = numerator
            # 设置分母（时值）
            if denominator == 4:
                header.timeSignature.denominator = Duration(value=Duration.quarter)
            elif denominator == 8:
                header.timeSignature.denominator = Duration(value=Duration.eighth)
            elif denominator == 2:
                header.timeSignature.denominator = Duration(value=Duration.half)
            else:
                header.timeSignature.denominator = Duration(value=Duration.quarter)

    def _process_track(self, song: gp.Song, track_data: Dict,
                       ticks_per_beat: int, track_index: int = 0,
                       time_signature: Tuple[int, int] = (4, 4)):
        """
        修复节拍问题的轨道处理方法
        """
        notes = track_data['notes']
        chords = track_data['chords']

        if not notes:
            return

        # 按时间排序所有事件（单音符和和弦）
        all_events = []
        for note in notes:
            if note['duration_ticks'] > 0:
                all_events.append(('note', note))
        '''for chord in chords:
            all_events.append(('chord', chord))'''

        all_events.sort(key=lambda x: x[1]['start_time'])

        # 处理所有事件
        processed_notes = set()
        previous_positions = []

        for event_type, event_data in all_events:
            start_time = event_data['start_time']

            # 如果已经处理过，跳过
            '''if event_type == 'note':
                note_id = (event_data['note'], start_time)
                if note_id in processed_notes:
                    continue
            else:  # chord
                chord_id = tuple(sorted(event_data['notes'])), start_time
                if chord_id in processed_notes:
                    continue'''

            measure_idx, beat_position = self._calculate_measure_and_beat(
                start_time, time_signature
            )

            midi_note = event_data['note']
            duration_ticks = event_data['duration_ticks']

            string, fret = self.find_best_position(midi_note, previous_positions)
            previous_positions.append((string, fret))
            if len(previous_positions) > 5:
                previous_positions.pop(0)

            duration_name = self.gp_generator.ticks_to_gp_duration(duration_ticks * Duration.quarterTime)

            success = self.gp_generator.add_note(song, string, fret, duration_name,
                                                 measure_index=measure_idx, position=beat_position)#, replace_existing=True)

            if not success:
                print(f"添加音符失败: MIDI音符{midi_note} -> {string}弦{fret}品")


    def _calculate_measure_and_beat(self, start_ticks: int,
                                    time_signature: Tuple[int, int]) -> Tuple[int, float]:
        """
        精确计算小节和拍子位置
        """
        numerator, _ = time_signature

        # 计算小节索引（从0开始）
        measure_idx = int(start_ticks // numerator)

        # 计算在小节中的拍子位置（从0开始）
        beat_in_measure = start_ticks % numerator

        assert beat_in_measure < numerator

        return measure_idx, beat_in_measure


class ChordDetector:
    """和弦检测器"""

    def __init__(self):
        self.chord_patterns = self._create_chord_patterns()

    def _create_chord_patterns(self) -> Dict[str, Set[int]]:
        return {
            'C': {0, 4, 7},
            'G': {7, 11, 2},
            'D': {2, 6, 9},
            'A': {9, 1, 4},
            'E': {4, 8, 11},
            'F': {5, 9, 0},
            'B': {11, 3, 6},
            'Cm': {0, 3, 7},
            'Gm': {7, 10, 2},
            'Dm': {2, 5, 9},
            'Am': {9, 0, 4},
            'Em': {4, 7, 11},
            'Bm': {11, 2, 6},
            'C7': {0, 4, 7, 10},
            'G7': {7, 11, 2, 5},
            'D7': {2, 6, 9, 0},
            'A7': {9, 1, 4, 7},
            'E7': {4, 8, 11, 2},
            'F7': {5, 9, 0, 3},
            'Cmaj7': {0, 4, 7, 11},
            'Gmaj7': {7, 11, 2, 6},
            'Cm7': {0, 3, 7, 10},
            'Gm7': {7, 10, 2, 5},
        }

    def detect_chord(self, midi_notes: List[int]) -> str:
        if len(midi_notes) < 2:
            return "Single"

        pitch_classes = set(note % 12 for note in midi_notes)

        best_match = None
        best_score = 0

        for chord_name, chord_pattern in self.chord_patterns.items():
            intersection = pitch_classes.intersection(chord_pattern)
            score = len(intersection)

            if score > best_score:
                best_score = score
                best_match = chord_name

        if best_match is None or best_score < 2:
            root_note = min(midi_notes) % 12
            root_names = {
                0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
            }
            return f"{root_names.get(root_note, 'Unknown')} Chord"

        return best_match


# 使用示例
def main():
    gp_generator = GuitarProGenerator()
    converter = MIDItoGP5Converter(gp_generator)

    midi_file = "0_cDRDBBZ7I.mid"
    output_file = "converted_fixed_timing.gp5"

    if os.path.exists(midi_file):
        # 尝试不同的拍号
        time_signatures = [(4, 4)]  # , (3, 4), (6, 8)]

        for ts in time_signatures:
            output = output_file.replace('.gp5', f'_{ts[0]}_{ts[1]}.gp5')
            converter.convert_midi_to_gp5(midi_file, output, time_signature=ts)
            print(f"使用拍号 {ts[0]}/{ts[1]} 转换完成")

    else:
        print(f"MIDI 文件不存在: {midi_file}")
        create_rhythmic_midi("rhythm_sample.mid")
        converter.convert_midi_to_gp5("rhythm_sample.mid", output_file, "Rhythm Guitar")


def create_rhythmic_midi(filename: str):
    """创建有节奏的示例 MIDI 文件"""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # 4/4 拍的节奏模式
    # 小节1: C 和弦 (四分音符)
    c_chord = [60, 64, 67]
    for note in c_chord:
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
    for note in c_chord:
        track.append(mido.Message('note_off', note=note, velocity=64, time=480))

    # 小节2: G 和弦 (八分音符)
    g_chord = [67, 71, 74]
    for note in g_chord:
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
    for note in g_chord:
        track.append(mido.Message('note_off', note=note, velocity=64, time=240))

    # 重复 G 和弦
    for note in g_chord:
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
    for note in g_chord:
        track.append(mido.Message('note_off', note=note, velocity=64, time=240))

    midi.save(filename)
    print(f"创建节奏示例 MIDI 文件: {filename}")


if __name__ == "__main__":
    main()
