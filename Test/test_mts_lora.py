from typing import Optional, Dict, List

import guitarpro as gp
import torch
import torchaudio
from peft import PeftModel

from model.dataset import decode
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen


def process_long_audio(
    model,
    audio_waveform: torch.Tensor,
    sample_rate: int,
    segment_duration: float = 8.0,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    处理任意长度的音频，自动分段并利用上下文生成连续的乐谱。

    Args:
        model: MTSGen模型实例
        audio_waveform: 音频波形，形状为 [1,1,T] 或 [T]（单声道）
        sample_rate: 采样率（Hz）
        segment_duration: 每段音频的最大时长（秒），默认8.0
        max_len: 模型能处理的最大序列长度（包括音频特征和上下文）。
                 若为None，则从模型配置或位置编码中自动获取。
        generate_length: 每段音频应生成的音符数量。若为None，则从模型配置
                         (notes_per_bar * predict_bars) 获取，默认64。
        device: 计算设备，若为None则使用模型参数所在设备。

    Returns:
        包含所有生成音符的字典，键为 'duration', 'fret', 'technique'，
        值为拼接后的张量，形状为 [1, total_notes, ...]。
    """
    # 确定设备
    if device is None:
        device = next(model.parameters()).device

    # 将音频转为标准形状 [batch=1, channels=1, time]
    if audio_waveform.dim() == 1:
        audio_waveform = audio_waveform.unsqueeze(0).unsqueeze(0)  # [1,1,T]
    elif audio_waveform.dim() == 2:
        # 假设为 [channels, time]，添加batch维度
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
        audio_waveform = audio_waveform.unsqueeze(0)               # [1,channels,time]
    audio_waveform = audio_waveform.float().to(device)

    # 获取模型最大序列长度限制
    if max_len is None:
        if hasattr(model.config, 'max_position_embeddings'):
            max_len = model.config.max_position_embeddings
        elif hasattr(model.positional_encoding, 'pe'):
            max_len = model.positional_encoding.pe.shape[1]
        else:
            max_len = 512  # 保守默认值
            print(f"Warning: Could not determine max_len, using default {max_len}")

    # 计算每段对应的样本数
    segment_samples = int(segment_duration * sample_rate)
    total_samples = audio_waveform.shape[-1]

    # 存储所有片段的生成结果（保持在CPU以节省显存）
    all_predictions: Dict[str, List[torch.Tensor]] = {
        'duration': [], 'fret': [], 'technique': []
    }

    # 历史上下文（CPU）
    context_notes: Optional[Dict[str, torch.Tensor]] = None

    model.eval()
    with torch.no_grad():
        start = 0
        while start < total_samples:
            end = min(start + segment_samples, total_samples)
            audio_segment = audio_waveform[..., start:end].to(device)

            # 准备上下文（截断至最大允许长度）
            if context_notes is not None:

                truncated_context = {}
                for key in ['duration', 'fret', 'technique']:
                    if key in context_notes:
                        context_notes[key].shape[1]
                        truncated = context_notes[key]
                        truncated_context[key] = truncated.to(device)
                context_to_use = truncated_context
            else:
                context_to_use = None

            # 生成当前段
            predictions = model.forward(
                audio_input=audio_segment,
                context_notes=context_to_use,
                teacher_forcing=False,
                generate_length=64,
                do_sample = True,
                return_logits=False
            )  # 返回字典，每个值形状 [1, generate_length, ...]

            # 将结果移至CPU并保存
            pred_cpu = {k: v.cpu() for k, v in predictions.items()}
            for key in all_predictions:
                all_predictions[key].append(pred_cpu[key])

            # 更新历史上下文（始终保留所有历史，下一段使用时再截断）
            if context_notes is None:
                context_notes = pred_cpu
            else:
                for key in context_notes:
                    context_notes[key] = torch.cat([context_notes[key], pred_cpu[key]], dim=1)

            start = end

    # 合并所有片段的结果
    merged = {}
    for key in all_predictions:
        merged[key] = torch.cat(all_predictions[key], dim=1)  # [1, total_notes, ...]

    return merged

config = MTSGenConfig.mtsGen_150m()
model = MTSGen(config)
model.load_state_dict(torch.load(f'checkpoint_epoch59.pth'))
model.to('cuda')
model = PeftModel.from_pretrained(model, "../output/Model/Lora")
wav, sr = torchaudio.load("又三郎.mp3")

if sr != 24000:
    resampler = torchaudio.transforms.Resample(sr, 24000)
    wav = resampler(wav)

result = process_long_audio(model, wav[:, :12 * 24000], 24000)
sample = {}
for key, value in result.items():
    if value is not None:
        sample[key] = value[0]

song = decode(sample)
gp.write(song, f'out.gp5')