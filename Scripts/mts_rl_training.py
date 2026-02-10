import logging
import random

import torch
from datasets import load_dataset

from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.rl.huggingface_dataset import IterableMIDIDataset
from model.rl.mid_comparitor import MidiVersionComparator
from model.rl.simulator import GuitarSequenceAnalyzer, PresetConfigs
from model.rl.trainer import RLTrainer, TestRLConfig

if __name__ == "__main__":
    seed = 43
    train_dataset = load_dataset("astune/mts_rl_dataset", streaming=True).filter(
        lambda x, idx: random.Random(seed + idx).random() > 0.1,
        with_indices=True
    )
    val_dataset = load_dataset("astune/mts_rl_dataset", streaming=True).filter(
        lambda x, idx: random.Random(seed + idx).random() <= 0.1,
        with_indices=True
    )

    training_dataset = IterableMIDIDataset(train_dataset['train'])
    val_dataset = IterableMIDIDataset(val_dataset['train'])


    config = MTSGenConfig.mtsGen_150m()
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch59.pth'))
    model.to('cuda')

    trainer_config = TestRLConfig()
    trainer_config.save_dir = "../output/Model/rl_training_results"
    trainer_config.num_epochs = 300
    trainer_config.log_interval = 20
    trainer_config.save_interval = 30
    trainer_config.batch_size = 4
    trainer_config.num_workers = 0

    trainer = RLTrainer(
        model=model,
        difficulty_system=GuitarSequenceAnalyzer(PresetConfigs.get_default()),
        similarity_system=MidiVersionComparator(),
        config=trainer_config,
    )
    trainer.logger.setLevel(logging.DEBUG)
    #trainer.load_checkpoint("")

    # 开始训练
    trainer.train(train_dataset=training_dataset, val_dataset = val_dataset)



