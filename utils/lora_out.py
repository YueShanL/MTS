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

    wav, sr = torchaudio.load(os.path.join(testSource, filename))

    device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

    #repo_id = "ylacombe/musicgen-melody-punk-lora"

    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody",
                                                                   torch_dtype=torch.float16).to(device)

    # Load LoRA adapter
    #model = PeftModel.from_pretrained(model, "../output/Lora/adaptor/checkpoint-epoch-10").to(device)
    # model = PeftModel.from_pretrained(model, repo_id).to(device)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")  # config.base_model_name_or_path)

    inputs = processor(
        text=["80s punk and pop track with bassy drums and synth", "80s blues track with groovy saxophone"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_values = audio_values.cpu().float().numpy()

    sf.write("musicgen_out_0.wav", audio_values[0].T, sampling_rate)
    sf.write("musicgen_out_1.wav", audio_values[1].T, sampling_rate)
