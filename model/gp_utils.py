from guitarpro import Duration

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

def duration_to_idx(duration: Duration):
    value = duration.value
    if duration.isDotted: value = value / 3 * 2
    for idx, dur in enumerate(duration_map):
        if dur[0] == 4 / value:
            return idx + 1
    return None

def idx_to_duration(idx):
    value, name, dotted = duration_map[idx]
    if dotted:
        value = value / 3 * 2
    return Duration(value=int(4 / value), isDotted = dotted)