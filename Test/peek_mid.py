import os
from pathlib import Path

import soundfile as sf

from data.loader import load_piast_dataset
from data.mid_preprocessor import midi_to_audio_tensor

dataset = load_piast_dataset(repo_path="../data/dataset/PIAST")
output_path = "./output/"

print(f'get dataset {dataset}')

subset = dataset['piast-yt']
midi_path = subset[1]['midi_path']

tensor, sr = midi_to_audio_tensor(midi_path)

audio_filename = os.path.join(output_path, f"{Path(midi_path).stem}_audio.wav")
sf.write(audio_filename, tensor, sr)
print(f"音频文件已保存: {audio_filename}")
