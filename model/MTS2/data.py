from dataclasses import dataclass
from typing import Dict, List, Any
import torch
import torchaudio

def encode_triplet(note: int, technique: int, duration: int, offset: int = 1) -> int:
    """
    将 (note, technique, duration) 编码为 token ID。
    参数：
        note: 0..25
        technique: 0..13
        duration: 0..12
        offset: 偏移量，用于保留特殊 token（如起始 token、padding token）
    返回：token_id = offset + note * (14*13) + technique * 13 + duration
    """
    assert 0 <= note < 26, f"note out of range: {note}"
    assert 0 <= technique < 14, f"technique out of range: {technique}"
    assert 0 <= duration < 13, f"duration out of range: {duration}"
    return offset + note * (14 * 13) + technique * 13 + duration

def decode_token(token_id: int, offset: int = 1) -> tuple:
    """
    将 token ID 解码为 (note, technique, duration)。
    """
    assert token_id >= offset, f"token_id {token_id} < offset {offset}"
    idx = token_id - offset
    note = idx // (14 * 13)
    remainder = idx % (14 * 13)
    technique = remainder // 13
    duration = remainder % 13
    return note, technique, duration

def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 处理音频（同之前）
    audio = torch.tensor(example["audio_input"], dtype=torch.float32).unsqueeze(0)
    orig_sr = 24000
    target_sr = 22050
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)

    # 2. 处理 target_notes
    notes = example["target_notes"]["fret"]          # list of list, shape (L, 6)
    techniques = example["target_notes"]["technique"]
    durations = example["target_notes"]["duration"]  # 假设每个轨道独立，形状 (L, 6)

    L = len(notes)
    num_heads = 6
    token_ids = torch.zeros((L, num_heads), dtype=torch.long)

    for t in range(L):
        for h in range(num_heads):
            note = notes[t][h]
            technique = techniques[t][h]
            duration = durations[t]

            # 范围校验
            assert 0 <= note < 26, f"note {note} out of range at step {t}, head {h}"
            assert 0 <= technique < 14, f"technique {technique} out of range at step {t}, head {h}"
            assert 0 <= duration < 13, f"duration {duration} out of range at step {t}, head {h}"

            token_id = encode_triplet(note, technique, duration, offset=1)  # 偏移 1，预留 0 给起始 token
            token_ids[t, h] = token_id

    # 3. 生成 input_ids（左移一位，开头插入起始 token）
    decoder_start_token_id = 0
    start_tokens = torch.full((1, num_heads), decoder_start_token_id, dtype=torch.long)
    input_ids = torch.cat([start_tokens, token_ids[:-1, :]], dim=0)  # [L, num_heads]
    labels = token_ids  # [L, num_heads]

    return {
        "waveform": audio,
        "input_ids": input_ids,
        "labels": labels,
    }


@dataclass
class DataCollatorForMTSGen2WithWaveform:
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取波形、input_ids、labels
        converted = [convert_example(f) for f in features]
        waveforms = [f["waveform"] for f in converted]
        input_ids = [f["input_ids"] for f in converted]
        labels = [f["labels"] for f in converted]

        # 波形填充：可能为 [C, T] 或 [T]，统一为 [C, T] 后填充
        # 这里假设波形是 [T] 或 [1, T]，我们确保形状为 [1, T] 并填充
        waveforms_padded = []
        for w in waveforms:
            if w.dim() == 1:
                w = w.unsqueeze(0)  # [1, T]
            waveforms_padded.append(w)
        waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms_padded, batch_first=True, padding_value=0.0)
        # 此时形状 [B, 1, max_T]

        # input_ids 和 labels 填充
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

        return {
            "waveform": waveforms_padded,
            "input_ids": input_ids,
            "labels": labels,
        }