from datasets import DatasetDict, Dataset
from transformers import AutoProcessor

from training.dataset import MusicGenMelodyDataset

dataset = DatasetDict.load_from_disk("../../output/Lora/dataset")

dataset = MusicGenMelodyDataset(
    Dataset.from_dict(dataset['train'][:2]),
    processor=AutoProcessor.from_pretrained('facebook/musicgen-melody')
)

print(len(dataset))
