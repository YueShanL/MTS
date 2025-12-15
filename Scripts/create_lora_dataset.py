from datasets import DatasetDict, Dataset, concatenate_datasets

from data.loader import load_piast_dataset
from training.dataset_builder import load_audio_dataset, process_style_transfer_dataset

debug = 0
#continue_on_exception = 1
linux = 0
if __name__ == '__main__':
    dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset/test"
    datadict_path = "output/Lora/datadict" if linux else "../output/Lora/datadict/test"
    generated_audio_dir = "output/Lora/training" if linux else "../output/Lora/training"

    piast = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

    data = load_audio_dataset(
        piast_data=piast['piast-yt'],
        generated_audio_dir=generated_audio_dir
    )

    def parse_generator():
        for example in process_style_transfer_dataset(Dataset.from_dict(data[:25]), generator=True):
            yield example

    # 创建迭代器
    #data_generator = process_style_transfer_dataset(Dataset.from_dict(data[:25]), generator=True)

    Dataset.from_generator(process_style_transfer_dataset, gen_kwargs={"dataset": Dataset.from_dict(data[:25]), "generator": True}).save_to_disk(dataset_path)

    # 分批处理
    batch_size = 20
    first_batch = True

    '''while True:
        try:
            # 收集一批数据
            batch = []
            for _ in range(batch_size):
                batch.append(next(data_generator))

            # 创建数据集
            batch_dataset = Dataset.from_list(batch)

            # 保存或追加
            if first_batch:
                batch_dataset.save_to_disk(dataset_path)
                first_batch = False
            else:
                existing_dataset = Dataset.load_from_disk(dataset_path)
                combined_dataset = concatenate_datasets([existing_dataset, batch_dataset])
                combined_dataset.save_to_disk(dataset_path)

            print(f"Processed batch with {len(batch)} records")

        except StopIteration:
            # 处理最后一批可能不满batch_size的数据
            if batch:
                existing_dataset = Dataset.load_from_disk(dataset_path)
                batch_dataset = Dataset.from_list(batch)
                combined_dataset = concatenate_datasets([existing_dataset, batch_dataset])
                combined_dataset.save_to_disk(dataset_path)
                print(f"Processed final batch with {len(batch)} records")
            break'''

    # 加载最终数据集
    dataset = Dataset.load_from_disk(dataset_path).with_format("torch")

    DatasetDict({"train": dataset}).save_to_disk(dataset_path)

    #data = process_style_transfer_dataset(Dataset.from_dict(data[:210]))