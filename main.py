import os
import soundfile as sf

import torch
import torchaudio
from transformers import MusicgenMelodyForConditionalGeneration, AutoProcessor

testSource = "./data/debug_output/"
filename = '7iPSSj62CUw_audio.wav'
modelname = "umxl"
'umxhq'


device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")

repo_id = "ylacombe/musicgen-melody-punk-lora"

model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody", torch_dtype=torch.float32).to(device)
#model = PeftModel.from_pretrained(model, repo_id).to(device)

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")#config.base_model_name_or_path)

wav, sr = torchaudio.load(os.path.join(testSource, filename))

inputs = processor(
    text=["80s punk and pop track with bassy drums and synth", "80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
).to(device)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().numpy()
sf.write("musicgen_out_0.wav", audio_values[0].T, sampling_rate)
sf.write("musicgen_out_1.wav", audio_values[1].T, sampling_rate)