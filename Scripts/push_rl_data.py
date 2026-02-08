import logging
import os

import torch
from datasets import load_dataset
from huggingface_hub import login, upload_file

from data.loader import load_piast_dataset
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.rl.dataset import MIDISegmentDataset
from model.rl.huggingface_dataset import create_hf_dataset_from_midi_segment
from model.rl.mid_comparitor import MidiVersionComparator
from model.rl.simulator import GuitarSequenceAnalyzer, PresetConfigs
from model.rl.trainer import RLTrainer, RLConfig, TestRLConfig

debug = 0
continue_on_exception = 1
linux = 0

if __name__ == "__main__":
    piast = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

    size = len(piast["piast-yt"])
    segment = 10
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    for idx in range(size // segment):

        dataset = MIDISegmentDataset(
            midi_dataset=piast["piast-yt"].select(range(idx * segment, (idx + 1) * segment)),
            batch=16,
            length_seconds=8,
            sample_rate=24000,
            overlap_ratio=0.3,
            shuffle=False,
            infinite=False,
            min_notes_per_segment=3,
            random_start=False
        )

        dataset = create_hf_dataset_from_midi_segment(dataset)

        dataset.to_parquet(os.path.join(current_dir, "/.ache/dataset.parquet"))

        upload_file(
            path_or_fileobj=os.path.join(current_dir, "/.ache/dataset.parquet"),
            path_in_repo=f"data/batch_{idx}.parquet",  # 指定存储路径和文件名
            repo_id="astune/mts_rl_dataset",
            repo_type="dataset",
            token=""  # 需要写权限
        )

