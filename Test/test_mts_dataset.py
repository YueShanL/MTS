import random

import guitarpro as gp

from datasets import Dataset
import soundfile as sf
from model.dataset import AudioGuitarTabDataset, decode

debug = 0
continue_on_exception = 1
linux = 0
if __name__ == '__main__':
    dataset_path = "../output/Model/dataset"
    output_path = "Test/output/dataset_samples/mts/" if linux else "output/dataset_samples/mts/"

    dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
    #dataset = AudioGuitarTabDataset(dataset['audio_input'], dataset['target_notes'])

    for i in range(10):
        sample = dataset[random.randint(0, len(dataset) - 1)]
        audio = sample['audio_input']
        target = decode(sample['target_notes'])
        context = decode(sample['context_notes'])
        gp.write(target, f'{output_path}target_{i}.gp5')
        gp.write(context, f'{output_path}context_{i}.gp5')
        sf.write(f'{output_path}audio_{i}.wav', audio.cpu().numpy(), 24000)
        print(i)