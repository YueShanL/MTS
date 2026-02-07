import logging

import torch
from huggingface_hub import login

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
    dataset = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

    dataset = MIDISegmentDataset(
        midi_dataset=dataset["piast-yt"],
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

    login("")

    dataset.push_to_hub("astune/mts_rl_dataset")

