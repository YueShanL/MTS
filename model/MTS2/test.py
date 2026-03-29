import torchaudio
from peft import PeftModel
from torch import vmap

from model.MTS2.data import decode_token
from model.MTS2.module import MTSGen2
from model.dataset import decode

import guitarpro as gp

model = MTSGen2.from_pretrained('../../output/Model/mts2/best').to('cuda')
model.to('cuda')
model = PeftModel.from_pretrained(model, "../../output/Model/Lora/mts2")
wav, sr = torchaudio.load("test.mp3")

if sr != 22050:
    resampler = torchaudio.transforms.Resample(sr, 22050)
    wav = resampler(wav).mean(dim=0, keepdim=False)[:20 * 22050]

# 2. 生成（默认起始 token ID = 0）
generated = model.generate(
    wav.to('cuda'),
    max_length=80,
)

fret, technique, duration = vmap(decode_token)(generated.to('cpu'))
duration = duration.to(float).mean(dim=2, keepdim=False).to(int)

song = decode({'fret': fret.squeeze(), 'technique': technique.squeeze(), 'duration': duration.squeeze()})

gp.write(song, f'lora_out.gp5')