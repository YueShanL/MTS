import os

import soundfile as sf
import torch
import torchaudio
from peft import PeftModel
from transformers import MusicgenMelodyForConditionalGeneration, AutoProcessor

if __name__ == "__main__":
    testSource = "../utils/debug_output/"
    filename = '7iPSSj62CUw_audio.wav'
    model_name = 'facebook/musicgen-melody'
    sample_rate = 32000

    # Check if GPU is available

    #wav, sr = torchaudio.load(os.path.join(testSource, filename))
    wav, sr = torchaudio.load("example.wav")

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        wav = resampler(wav)

    device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody",
                                                                   torch_dtype=torch.float16).to(device)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "../output/Lora/adaptor/musicgen_lora_piano_large_dataset").to(device)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")  # config.base_model_name_or_path)

    inputs = processor(
        audio = wav.squeeze()[:5*32000],
        sampling_rate = sample_rate,
        text=["piano cover"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    inputs.data['input_features']=inputs.data['input_features'].to(torch.float16)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_values = audio_values.cpu().float().numpy()

    sf.write("musicgen_out_0.wav", audio_values[0].T, sampling_rate)
    #sf.write("musicgen_out_1.wav", audio_values[1].T, sampling_rate)
