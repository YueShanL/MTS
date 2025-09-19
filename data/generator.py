import os
import random

import pretty_midi
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

from data.mid_preprocessor import midi_to_audio_tensor

styles = ['Pop', 'Synth-pop', 'Dance Pop', 'Pop Rock', 'Electropop', 'Hip-Hop', 'Rap', 'Boom-Bap', 'Trap', 'Jazz Rap',
          'Drill', 'Emo Rap', 'Rock', 'Punk Rock', 'Alternative Rock', 'Indie Rock', 'Hard Rock', 'Electronic', 'EDM',
          'House', 'Techno', 'Trance', 'Dubstep', 'Drum & Bass', 'R&B', 'Soul', 'Contemporary R&B', 'Neo-Soul', 'Funk',
          'Country', 'Traditional Country', 'Pop Country', 'Country Rock', 'Jazz', 'Swing', 'Bebop', 'Cool Jazz',
          'Fusion',
          'Classical', 'Baroque', 'Romantic', 'Modern Classical', 'Metal', 'Heavy Metal', 'Death Metal', 'Black Metal',
          'Metalcore', 'Folk', 'Traditional Folk', 'Contemporary Folk', 'Folk Rock', 'Latin', 'Reggaeton', 'Salsa',
          'Bachata', 'Latin Pop', 'K-Pop', 'Blues', 'Delta Blues', 'Chicago Blues', 'Electric Blues', 'World']
a_model = MusicGen.get_pretrained('facebook/musicgen-melody-large')


def generate(source_path, tag: str, output_path, repeating_limit=1, model=a_model, time_limit=-1):
    file_name = os.path.basename(source_path)
    i = 0
    for f in os.listdir(output_path):
        if file_name in f:
            i += 1
        if i >= repeating_limit:
            return

    while i < repeating_limit:
        i += 1

        midi_data = pretty_midi.PrettyMIDI(source_path)
        duration = midi_data.get_end_time() if time_limit == -1 else min(float(time_limit), midi_data.get_end_time())
        audio_tensor, sr = midi_to_audio_tensor(
            source_path,
            debug=False,
            save_audio=False,
            visualize=False,
        )

        model.set_generation_params(duration=duration)  # generate 8 seconds.
        style = random.choice(styles)
        while f'{file_name}_{style}.wav' in os.listdir(output_path):
            st = style
            style = random.choice(styles)
            print(f'{file_name}_{st}.wav already exist, trying {style}')
        prompt = style + ", " + tag
        # generates using the melody from the given audio and the provided descriptions.
        wav, token = model.generate_with_chroma([prompt], audio_tensor[None].expand(1, -1, -1), sr, return_tokens=True)
        audio_write(f'{output_path}/{file_name}_{style}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
