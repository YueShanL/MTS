import os
import random

import soundfile as sf
from datasets import DatasetDict

dataset = DatasetDict.load_from_disk("../../output/Lora/dataset")
output_path = "./output/"

print(f'get dataset {dataset}')

subset = random.choice(dataset['train'])


tensor = subset["target_audio_values"]
sr = 32000

audio_filename = os.path.join(output_path, f"debug_audio.wav")
sf.write(audio_filename, tensor, sr)
print(f"音频文件已保存: {audio_filename}")