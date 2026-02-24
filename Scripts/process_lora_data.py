from pathlib import Path

import guitarpro as gp
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import MusicgenMelodyForConditionalGeneration, AutoProcessor

from model.dataset import AudioGuitarTabDataset
from model.rl.mid_comparitor import midi_to_pretty_midi
from utils.GP5Generator import GuitarProGenerator
from utils.gp2mid import gp5_to_midi
from utils.mid2gp import MIDItoGP5Converter
from utils.mid_preprocessor import midi_to_audio_tensor

device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")



def preprocessing_tabs(path, model, processor, segment_duration=8, overlap=0.3):
    # 可调参数
    batch_size = 2
    sample_rate = 32000
    extensions = ['.gp5']
    midi_files = []

    for ext in extensions:
        midi_files.extend(Path(path).rglob(f'*{ext}'))

    midi_files = sorted(list(set(midi_files)))

    print(f"find {len(midi_files)} tabs")

    audio_inputs = []
    tab_data = []

    guitar_dataset = AudioGuitarTabDataset([], [])
    converter = MIDItoGP5Converter(GuitarProGenerator())
    for m in midi_files:
        song = gp.parse(m)
        audio_tensor, _ = midi_to_audio_tensor(
            midi_to_pretty_midi(gp5_to_midi(song, output_midi_path=None, tempo=120)), sr=sample_rate)
        converter.quantize_song(song)
        print("----")
        tab_dict = guitar_dataset.encode_tab_sequence(song)  # {fret:[len, 6], technique:[len, 6], duration:[len]}
        L = audio_tensor.shape[-1] // sample_rate
        bps = 8  # len(tab_dict['duration']) / L

        # 计算切片相关参数
        hop_len_steps = int(segment_duration * (1 - overlap))  # 滑动步长（步数）
        if hop_len_steps == 0:
            hop_len_steps = 1  # 防止死循环，确保至少步进1步

        # 计算可生成的切片数量（仅考虑完整切片）
        num_segments = (L - segment_duration) // hop_len_steps + 1

        # 生成切片
        for i in range(num_segments):
            start_sec = i * hop_len_steps
            end_sec = start_sec + segment_duration

            # 截取 tab 序列片段
            tab_segment = {key: value[int(start_sec * bps):int(end_sec * bps)] for key, value in tab_dict.items()}

            # 截取音频片段
            start_sample = start_sec * sample_rate
            end_sample = end_sec * sample_rate
            audio_segment = audio_tensor[start_sample:end_sample]

            tab_data.append(tab_segment)
            audio_inputs.append(audio_segment)

    audio_out = []
    for i in range(len(tab_data) // batch_size):
        batch = audio_inputs[i * batch_size:(i + 1) * batch_size]
        inputs = processor(
            text=["piano cover"] * len(batch),
            audio=batch,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
        ).to(device)
        segment = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=410).squeeze(1)[
            :, :segment_duration * sample_rate].cpu().float().numpy()
        audio_out.extend(segment)

    return audio_out, tab_data


musicGen_model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody",
                                                                        torch_dtype=torch.float32).to(device)
# Load LoRA adapter
musicGen_model = PeftModel.from_pretrained(musicGen_model,
                                           "../output/Lora/adaptor/musicgen_lora_piano_large_dataset").to(device)

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")  # config.base_model_name_or_path)

audio_inputs, target_notes = preprocessing_tabs("../data/tabs", musicGen_model, processor)

dataset = Dataset.from_list([{'audio_inputs': a, 'target_notes': n} for a, n in zip(audio_inputs, target_notes)])
dataset.save_to_disk("mts_lora_dataset")