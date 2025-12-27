import guitarpro
from guitarpro import Duration, Song
from mido import Message, MidiFile, MidiTrack, MetaMessage


def gp5_to_midi_simple(gp5_file, output_midi_path='output.mid',
                       track_index=0, program=25, ticks_per_beat=480, tempo=120):
    """
    简化版GP5到MIDI转换函数，修复同一时刻音符拆分问题

    Args:
        gp5_file: GP5文件路径
        output_midi_path: 输出MIDI文件路径
        track_index: 要转换的音轨索引
        program: MIDI音色程序号
        ticks_per_beat: MIDI每拍ticks数
        tempo: 默认速度（BPM）
    """
    try:
        # 1. 解析GP5文件
        if isinstance(gp5_file, str):
            song = guitarpro.parse(gp5_file)
        elif isinstance(gp5_file, Song):
            song = gp5_file
        else:
            raise TypeError("gp5_file must be str or Song")

            # 2. 创建MIDI文件
        midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)

        # 3. 设置基本MIDI信息
        track_name = song.tracks[track_index].name if track_index < len(song.tracks) else "GP5 Track"
        midi_track.append(MetaMessage('track_name', name=track_name, time=0))

        # 设置音色
        midi_track.append(Message('program_change', program=program, time=0))

        # 设置速度
        actual_tempo = song.tempo if hasattr(song, 'tempo') else tempo
        from mido import bpm2tempo
        tempo_midi = bpm2tempo(actual_tempo)
        midi_track.append(MetaMessage('set_tempo', tempo=tempo_midi, time=0))

        # 4. 获取GP音轨
        if track_index >= len(song.tracks):
            print(f"错误: 音轨索引 {track_index} 超出范围 (0-{len(song.tracks) - 1})")
            return None

        gp_track = song.tracks[track_index]

        # 5. 处理音轨中的所有音符
        process_track_notes(gp_track, midi_track, ticks_per_beat)

        # 6. 添加轨道结束标记
        midi_track.append(MetaMessage('end_of_track', time=0))

        # 7. 保存MIDI文件
        midi_file.save(output_midi_path)
        print(f"转换完成: {output_midi_path}")

        return midi_file

    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_track_notes(gp_track, midi_track, ticks_per_beat):
    """
    处理音轨中的音符，正确处理延音线
    """
    # 收集所有音符（已处理延音线）
    notes_list = collect_notes_with_tie_processing(gp_track, ticks_per_beat)

    # 按时间排序
    notes_list.sort(key=lambda x: x['start_time'])

    # 准备事件列表
    all_events = []
    for note_data in notes_list:
        # Note On事件
        all_events.append((
            note_data['start_time'],
            'note_on',
            note_data['note'],
            note_data['velocity']
        ))

        # Note Off事件
        all_events.append((
            note_data['start_time'] + note_data['duration'],
            'note_off',
            note_data['note'],
            0
        ))

    # 排序并添加到MIDI轨道
    all_events.sort(key=lambda x: x[0])
    last_time = 0

    for time, event_type, note, velocity in all_events:
        delta_time = time - last_time

        if event_type == 'note_on':
            midi_track.append(Message('note_on', note=note, velocity=velocity, time=delta_time))
        elif event_type == 'note_off':
            midi_track.append(Message('note_off', note=note, velocity=velocity, time=delta_time))

        last_time = time

    return len(notes_list)


def calculate_duration_simple(duration, ticks_per_beat):
    """
    简化版时长计算
    """
    base_durations = {
        1: 4.0,  # 全音符
        2: 2.0,  # 二分音符
        4: 1.0,  # 四分音符
        8: 0.5,  # 八分音符
        16: 0.25,  # 十六分音符
        32: 0.125,  # 三十二分音符
        64: 0.0625  # 六十四分音符
    }

    # 获取duration的value属性
    duration_value = duration.value if hasattr(duration, 'value') else 4

    beats = base_durations.get(duration_value, 1.0)

    # 处理附点
    if hasattr(duration, 'isDotted') and duration.isDotted:
        beats *= 1.5

    return int(beats * ticks_per_beat)


def calculate_note_pitch_simple(note):
    """
    简化版音高计算，只考虑基础音高和基本技巧
    """
    # 获取基础MIDI音高
    if hasattr(note, 'realValue'):
        base_note = note.realValue
    elif hasattr(note, 'value') and hasattr(note, 'string'):
        # 备用方法计算音高
        base_note = note.value
        if hasattr(note, 'string'):
            # 吉他标准调弦
            string_tuning = [64, 59, 55, 50, 45, 40]  # 从1弦到6弦
            if 1 <= note.string <= len(string_tuning):
                base_note += string_tuning[note.string - 1]
    else:
        base_note = 60

    final_note = float(base_note)

    # 只处理推弦（对音高影响最明显的技巧）
    if hasattr(note.effect, 'bend') and note.effect.bend:
        bend = note.effect.bend
        if hasattr(bend, 'points') and bend.points:
            # 获取最大推弦值
            max_bend = 0
            for point in bend.points:
                if hasattr(point, 'value'):
                    max_bend = max(max_bend, point.value)

            # 简单转换为半音
            if max_bend > 0:
                semitone_bend = max_bend / 100.0  # 假设100为全音
                final_note += semitone_bend

    # 处理泛音（对音高影响明显的技巧）
    if hasattr(note.effect, 'harmonic') and note.effect.harmonic:
        final_note += 12  # 简单处理：泛音升高一个八度

    # 确保音高在有效范围内
    final_note = max(0, min(127, int(round(final_note))))

    return final_note


def find_tie_duration(note, start_beat, start_time, ticks_per_beat):
    """
    查找延音线的持续时间

    Args:
        note: 起始音符
        start_beat: 起始拍子
        start_time: 起始时间
        ticks_per_beat: 每拍ticks数

    Returns:
        int: 延音线总时长（ticks）
    """
    # 简化的延音线查找：在当前拍子范围内查找相同音高的音符
    # 实际实现需要遍历后续拍子
    total_duration = 0

    # 获取当前音符的音高信息
    note_value = note.value
    note_string = note.string

    # 查找当前拍子内的延音线
    current_beat = start_beat
    beat_duration = calculate_duration_simple(start_beat.duration, ticks_per_beat)
    total_duration += beat_duration

    # 注意：实际实现需要遍历整个音轨查找延音线
    # 这里返回基础时长作为占位
    return total_duration


def process_note_with_tie(note, beat, start_time, active_notes, ticks_per_beat):
    """
    处理音符，正确处理延音线效果

    Args:
        note: 音符对象
        beat: 拍子对象
        start_time: 起始时间（绝对ticks）
        active_notes: 活跃的延音线字典 {note_id: [start_time, duration_so_far]}
        ticks_per_beat: 每拍ticks数

    Returns:
        dict: 处理后的音符数据（可能为None如果只是延音线延续）
        dict: 更新后的活跃延音线字典
    """
    # 生成音符唯一标识（弦+品位）
    note_id = f"{note.string}_{note.value}"

    # 检查音符类型
    if note.type == guitarpro.NoteType.tie:
        # 延音线音符 - 不创建新音符，只延长已有音符

        if active_notes.get(note.string):
            active_notes[note.string]['duration'] += calculate_duration_simple(beat.duration, ticks_per_beat)
            return None
        else:
            # 延音线开始处没有找到原始音符，可能是文件有问题
            # 当作普通音符处理
            print(f"警告: 延音线音符 {note_id} 没有找到起始音符")

    # 普通音符处理
    # 计算音高
    midi_note = calculate_note_pitch_simple(note)

    # 计算基础时长
    base_duration = calculate_duration_simple(beat.duration, ticks_per_beat)

    # 创建音符数据
    note_data = {
        'start_time': start_time,
        'note': midi_note,
        'velocity': note.velocity,
        'duration': base_duration,
        'note_id': note_id,
    }

    return note_data


def collect_notes_with_tie_processing(gp_track, ticks_per_beat):
    """
    收集音轨中的所有音符，正确处理延音线

    Args:
        gp_track: GP音轨对象
        ticks_per_beat: 每拍ticks数

    Returns:
        list: 处理后的音符列表
    """
    notes_list = []
    active_tied_notes = {}  # 活跃的延音线 {note_id: {start_time, duration}}

    # 遍历所有小节、声部和拍子
    for measure in gp_track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                beat_start_time = int(beat.start / Duration.quarterTime * ticks_per_beat) # gp tick to beat time to mid tick

                # 跳过休止符
                if beat.status == guitarpro.BeatStatus.rest:
                    continue

                # 处理当前拍子中的所有音符
                for note in beat.notes:
                    if note.type == guitarpro.NoteType.rest:
                        continue

                    # 处理音符，考虑延音线
                    note_data = process_note_with_tie(
                        note, beat, beat_start_time, active_tied_notes,
                        ticks_per_beat
                    )

                    # 如果有音符数据（非延音线延续），添加到列表
                    if note_data:
                        notes_list.append(note_data)
                        active_tied_notes[note.string] = note_data

    return notes_list

# 使用示例
if __name__ == "__main__":
    # 转换单个音轨
    midi_file = gp5_to_midi_simple('../model/out.gp5', '../model/out.mid', track_index=0)
    gp5_to_midi_simple('../model/target.gp5', '../model/target.mid', track_index=0)
    if midi_file:
        print(f"MIDI文件信息:")
        print(f"  音轨数: {len(midi_file.tracks)}")
        print(f"  文件长度: {midi_file.length:.2f} 秒")

        # 统计音符数量
        note_count = 0
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'note_on':
                    note_count += 1
        print(f"  总音符数: {note_count}")