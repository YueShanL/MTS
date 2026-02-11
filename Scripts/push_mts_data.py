import os
import sys

from huggingface_hub import upload_file

sys.path.append('/data/projects/punim2072/MTS/MTS-Steve/MTS')

from datasets import Dataset, IterableDataset

from model.dataset import AudioGuitarTabDataset

linux = 1
debug = 0
token = ""
if __name__ == '__main__':
    dataset_path = "output/Model/dataset" if linux else "../output/Model/dataset"
    output_path = "output/Model/tf_50m" if linux else "../output/Model"
    piast_yt = "data/dataset/PIAST/piast_yt/midi" if linux else "../data/dataset/PIAST/piast_yt/midi"

    iterable_dataset = IterableDataset.from_generator(
        AudioGuitarTabDataset.generator_from_path,
        gen_kwargs={
            "mid_path": piast_yt,
            "start": 0,
            "type": 'py'
        })

    temp_dir = f"{dataset_path}_temp"
    batch_size = 10000
    batch_count = 0
    current_batch = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    for record in iterable_dataset:
        if len(current_batch) >= batch_size:
            # 保存当前批次
            batch_dataset = Dataset.from_list(current_batch)

            batch_dataset.to_parquet(os.path.join(current_dir, "dataset.parquet"))

            upload_file(
                path_or_fileobj=os.path.join(current_dir, "dataset.parquet"),
                path_in_repo=f"data/batch_{batch_count}.parquet",  # 指定存储路径和文件名
                repo_id="astune/mts_dataset",
                repo_type="dataset",
                token=token  # 需要写权限
            )

            batch_count += 1
            current_batch = []
            print(f"Saved batch {batch_count}")
        current_batch.append(record)

    # 保存最后一批
    if current_batch:
        batch_dataset = Dataset.from_list(current_batch)
        batch_dataset.to_parquet(os.path.join(current_dir, "dataset.parquet"))

        upload_file(
            path_or_fileobj=os.path.join(current_dir, "dataset.parquet"),
            path_in_repo=f"data/batch_{batch_count}.parquet",  # 指定存储路径和文件名
            repo_id="astune/mts_dataset",
            repo_type="dataset",
            token=token  # 需要写权限
        )