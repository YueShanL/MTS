import random

from datasets import Features, Value, Sequence, load_dataset, Dataset

from model.dataset import AudioGuitarTabDataset
from model.mts_generate import MTSGenTrainer, MTSGenConfig

linux = 0
debug = 0
if __name__ == '__main__':
    dataset_path = "output/Model/dataset" if linux else "../output/Model/dataset"
    output_path = "output/Model" if linux else "../output/Model"

    piast_yt = "data/dataset/PIAST/piast_yt/midi" if linux else "../data/dataset/PIAST/piast_yt/midi"

    generating_dataset = False

    try:
        dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
        dataset = AudioGuitarTabDataset(dataset['audio_input'], dataset['target_notes'])
    except Exception as e:
        print(f'failed to load from {dataset_path} because {e}, trying to generate dataset')
        generating_dataset = True

    if generating_dataset:
        dataset, data = AudioGuitarTabDataset.create_from_path(piast_yt, limit = 200)
        Dataset.from_dict(data).save_to_disk(dataset_path=dataset_path)

    #s = dataset[random.randint(0, len(dataset) - 1)]
    #print(s)

    config = MTSGenConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_durations=13,
        num_techniques=14,
        context_bars=4,
        predict_bars=1,
        max_fret=24,
        freeze_encoder=True
    )
    trainer = MTSGenTrainer(config)
    trainer.train(dataset, output_path=output_path)