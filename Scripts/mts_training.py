import torch
from datasets import Dataset, IterableDataset

from model.dataset import AudioGuitarTabDataset
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGenTrainer, MTSGen

linux = 0
debug = 0
if __name__ == '__main__':
    dataset_path = "output/Model/dataset" if linux else "../output/Model/dataset"
    output_path = "output/Model" if linux else "../output/Model"
    piast_yt = "data/dataset/PIAST/piast_yt/midi" if linux else "../data/dataset/PIAST/piast_yt/midi"

    generating_dataset = False

    try:
        dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
        #dataset = AudioGuitarTabDataset(dataset['audio_input'], dataset['target_notes'])
    except Exception as e:
        print(f'failed to load from {dataset_path} because {e}, trying to generate dataset')
        generating_dataset = True

    if generating_dataset:
        dataset, _ = AudioGuitarTabDataset.create_from_path(piast_yt, limit = 1000)
        Dataset.from_generator(dataset.stream_generator).save_to_disk(dataset_path=dataset_path)
        dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
    #Dataset.from_generator(dataset.stream_generator).save_to_disk(dataset_path=dataset_path)


    config = MTSGenConfig.mtsGen_30m()
    model = MTSGen(config)
    #model.load_state_dict(torch.load(f'checkpoint_epoch_20.pt')['model_state_dict'])
    trainer = MTSGenTrainer(config, model = model)
    trainer.train(dataset, output_path=output_path)
