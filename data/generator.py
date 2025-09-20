import os
import random

import pretty_midi
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

from data.mid_preprocessor import midi_to_audio_tensor

styles = ['Pop', 'Synth-pop', 'Dance Pop', 'Pop Rock', 'Electropop', 'Hip-Hop', 'Rap', 'Boom-Bap', 'Trap', 'Jazz Rap',
          'Drill', 'Emo Rap', 'Rock', 'Punk Rock', 'Alternative Rock', 'Indie Rock', 'Hard Rock', 'Electronic', 'EDM',
          'House', 'Techno', 'Trance', 'Dubstep', 'J-Pop', 'R&B', 'Soul', 'Contemporary R&B', 'Neo-Soul', 'Funk',
          'Country', 'Traditional Country', 'Pop Country', 'Country Rock', 'Jazz', 'Swing', 'Bebop', 'Cool Jazz',
          'Fusion',
          'Classical', 'Baroque', 'Romantic', 'Modern Classical', 'Metal', 'Heavy Metal', 'Death Metal', 'Black Metal',
          'Metalcore', 'Folk', 'Traditional Folk', 'Contemporary Folk', 'Folk Rock', 'Latin', 'Reggaeton', 'Salsa',
          'Bachata', 'Latin Pop', 'K-Pop', 'Blues', 'Delta Blues', 'Chicago Blues', 'Electric Blues', 'World']
a_model = MusicGen.get_pretrained('facebook/musicgen-melody-large')


def generate(source_path, tag: str, output_path, fix_style=None, repeating_limit=1, model=a_model, time_limit=-1, split_audio=0):
    file_name = os.path.splitext(os.path.basename(source_path))[0]
    i = 0
    for f in os.listdir(output_path):
        if file_name in f:
            i += 1
        if i >= repeating_limit:
            print(f'{file_name} has processed {repeating_limit} times, skipped to next')
            return

    while i < repeating_limit:
        i += 1

        midi_data = pretty_midi.PrettyMIDI(source_path)
        duration = midi_data.get_end_time() if time_limit == -1 else min(float(time_limit), midi_data.get_end_time())
        print(f'processing {file_name}-{duration}s...')

        audio_tensor, sr = midi_to_audio_tensor(
            source_path,
            debug=False,
            save_audio=False,
            visualize=False,
        )

        audio_segments = []
        if split_audio:
            segment_samples = int(time_limit * sr)
            total_samples = len(audio_tensor)
            num_segments = min(total_samples // segment_samples, split_audio)

            for i in range(num_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                segment = audio_tensor[start_idx:end_idx]
                audio_segments.append(segment.expand(1, -1))

            print(f"split into {len(audio_segments)} audios")
        else:
            audio_tensor = audio_tensor[:int(sr*duration)]

        model.set_generation_params(duration=duration)  # generate 8 seconds.
        style = random.choice(styles)
        if not fix_style:
            while f'{file_name}_{style}.wav' in os.listdir(output_path):
                st = style
                style = random.choice(styles)
                print(f'{file_name}_{st}.wav already exist, trying {style}')
        else:
            style = fix_style
        prompt = f'{style}, {tag}'
        # generates using the melody from the given audio and the provided descriptions.
        if split_audio:
            wav = model.generate_with_chroma([prompt] * len(audio_segments), audio_segments, sr)
            for idx, w in enumerate(wav):
                audio_write(f'{output_path}/{file_name}_part{idx}_{style}', w[0].cpu(), model.sample_rate, strategy="loudness",
                            loudness_compressor=True)
        else:
            wav, token = model.generate_with_chroma([prompt], audio_tensor.expand(1, -1, -1), sr, return_tokens=True)
            audio_write(f'{output_path}/{file_name}_{style}', wav[0].cpu(), model.sample_rate, strategy="loudness",
                        loudness_compressor=True)
