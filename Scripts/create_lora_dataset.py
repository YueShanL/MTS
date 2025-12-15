from datasets import DatasetDict, Dataset

from data.loader import load_piast_dataset
from training.dataset_builder import load_audio_dataset, process_style_transfer_dataset

debug = 0
#continue_on_exception = 1
linux = 0
if __name__ == '__main__':
    dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset/test"
    generated_audio_dir = "output/Lora/training" if linux else "../output/Lora/training"
    output_path = "output/Lora/adaptor/checkpoints/" if linux else "../output/Lora/adaptor/checkpoints/"

    piast = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

    data = load_audio_dataset(
        piast_data=piast['piast-yt'],
        generated_audio_dir=generated_audio_dir
    )

    dataset = Dataset.from_generator(process_style_transfer_dataset, gen_kwargs={"dataset": Dataset.from_dict(data[:400]), "generator": True})
    DatasetDict({"train": dataset}).save_to_disk(dataset_path)

    #data = process_style_transfer_dataset(Dataset.from_dict(data[:210]))

    print(data.info)
    DatasetDict({"train": data}).save_to_disk(dataset_path)
    data = data.to_iterable_dataset().filter(
        lambda example, idx: len(example['input_audio_values']) > 0, with_indices=True)
