import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


testSource = "./asset/"
filename = 'songTest.mp3'
modelname = "umxl"
'umxhq'

model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
model.set_generation_params(duration=30, cfg_coef=3)  # generate 8 seconds.
#wav = model.generate_unconditional(1)    # generates 4 unconditional audio samples
#descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
#wav = model.generate(['guitar finger-style'])#descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('umxl.songTest.mp3.vocals.wav')

# generates using the melody from the given audio and the provided descriptions.
wav, token = model.generate_with_chroma(['solo,piano'], melody[None].expand(1, -1, -1), sr, return_tokens=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)