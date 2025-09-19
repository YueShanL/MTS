import os

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


testSource = "./data/debug_output/"
filename = '7iPSSj62CUw_audio.wav'
modelname = "umxl"
'umxhq'


model = MusicGen.get_pretrained('facebook/musicgen-melody-large')
model.set_generation_params(duration=30, cfg_coef=3)  # generate 8 seconds.
#wav = model.generate_unconditional(1)    # generates 4 unconditional audio samples
#descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
#wav = model.generate(['guitar finger-style'])#descriptions)  # generates 3 samples.

melody, sr = torchaudio.load(os.path.join(testSource, filename))
prompt = ['solo, piano cover, rearrange']
# generates using the melody from the given audio and the provided descriptions.
wav, token = model.generate_with_chroma(['Jpop'], melody[None].expand(1, -1, -1), sr, return_tokens=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)