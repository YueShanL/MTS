import os
import shutil

from datasets import DatasetDict, Dataset, concatenate_datasets, IterableDataset

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

    temp_dir = f"{dataset_path}_cache"

    iterable_dataset = IterableDataset.from_generator(process_style_transfer_dataset, gen_kwargs={"dataset": Dataset.from_dict(data[:400]), "generator": True})
    # 分批处理
    batch_size = 20
    batch_count = 0
    first_batch = True

    current_batch = []

    for record in iterable_dataset:
        current_batch.append(record)

        if len(current_batch) >= batch_size:
            # 保存当前批次
            batch_path = os.path.join(temp_dir, f"batch_{batch_count}")
            os.makedirs(batch_path, exist_ok=True)

            batch_dataset = Dataset.from_list(current_batch)
            batch_dataset.save_to_disk(batch_path, num_shards=batch_size - 1)

            batch_count += 1
            current_batch = []
            print(f"Saved batch {batch_count}")

    # 保存最后一批
    if current_batch:
        batch_path = os.path.join(temp_dir, f"batch_{batch_count}")
        os.makedirs(batch_path, exist_ok=True)
        batch_dataset = Dataset.from_list(current_batch)
        batch_dataset.save_to_disk(batch_path, num_shards=len(current_batch) - 1)

    # 合并所有批次
    if os.path.exists(temp_dir):
        # 列出所有批次
        batch_dirs = sorted([d for d in os.listdir(temp_dir)
                           if os.path.isdir(os.path.join(temp_dir, d))])

        if batch_dirs:
            # 加载第一个批次
            first_batch_path = os.path.join(temp_dir, batch_dirs[0])
            combined_dataset = Dataset.load_from_disk(first_batch_path)

            # 合并其他批次
            for batch_dir in batch_dirs[1:]:
                batch_path = os.path.join(temp_dir, batch_dir)
                batch_dataset = Dataset.load_from_disk(batch_path)
                combined_dataset = concatenate_datasets([combined_dataset, batch_dataset])

            # 保存合并后的数据集
            combined_dataset.save_to_disk(dataset_path)

            # 清理临时目录
            shutil.rmtree(temp_dir)

    # 加载最终数据集
    dataset = Dataset.load_from_disk(dataset_path).with_format("torch")

    DatasetDict({"train": dataset}).save_to_disk(dataset_path)

    #data = process_style_transfer_dataset(Dataset.from_dict(data[:210]))