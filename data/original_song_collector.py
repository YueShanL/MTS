from data_collection import YouTubePianoCoverDataset
from utils import config

path = "piano_covers_dataset600.csv"

dataset = YouTubePianoCoverDataset(config.getAPIValue('youtube'))
dataset.find_original_songs(csv_file_path=path)#, top_n=10)
