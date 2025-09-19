from data.data_collection import YouTubePianoCoverDataset
from utils import config


path = "piano_covers_dataset600.csv"

dataset = YouTubePianoCoverDataset(config.getAPIValue('youtube'), config.getAPIValue('LLM'))
dataset.extract_original_title(csv_file_path=path, overwrite=True)