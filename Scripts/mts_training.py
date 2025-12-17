import os
import shutil

import torch
from datasets import Dataset, IterableDataset, concatenate_datasets

from model.dataset import AudioGuitarTabDataset
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.trainer import train_mixed_model

linux = 0
debug = 0
if __name__ == '__main__':
    dataset_path = "output/Model/dataset" if linux else "../output/Model/dataset"
    output_path = "output/Model" if linux else "../output/Model"
    piast_yt = "data/dataset/PIAST/piast_yt/midi" if linux else "../data/dataset/PIAST/piast_yt/midi"

    generating_dataset = False
    dataset_length = 400
    current_length = 0

    try:
        dataset = Dataset.load_from_disk(dataset_path)
        current_length = int(dataset.info.description)
        if current_length < dataset_length:
            generating_dataset = True
        #dataset = AudioGuitarTabDataset(dataset['audio_input'], dataset['target_notes'])
    except Exception as e:
        print(f'failed to load from {dataset_path} because {e}, trying to generate dataset')
        generating_dataset = True

    if generating_dataset:
        iterable_dataset = IterableDataset.from_generator(
            AudioGuitarTabDataset.generator_from_path,
            gen_kwargs={
                "mid_path": piast_yt,
                "start": current_length,
                "limit" : dataset_length - current_length
            })

        temp_dir = f"{dataset_path}_temp"
        batch_size = 500
        batch_count = 0
        current_batch = []

        if current_length >= 0:
            current_batch = dataset.to_list()
            print(f"Found existing dataset with {current_length} samples")

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
        dataset = Dataset.load_from_disk(dataset_path)
    #Dataset.from_generator(dataset.stream_generator).save_to_disk(dataset_path=dataset_path)


    config = MTSGenConfig.mtsGen_150m()
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch_20.pt')['model_state_dict'])
    model.to('cuda')
    train_mixed_model(model, dataset, val_dataset=None,
                          num_epochs=20, batch_size=8)
