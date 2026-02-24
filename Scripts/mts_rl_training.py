import logging
import random

import torch
from datasets import load_dataset, Dataset

from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.rl.huggingface_dataset import IterableMIDIDataset
from model.rl.mid_comparitor import MidiVersionComparator
from model.rl.simulator import GuitarSequenceAnalyzer, PresetConfigs
from model.rl.trainer import RLTrainer, TestRLConfig

if __name__ == "__main__":
    seed = 43
    val_size = 6
    train_dataset = Dataset.from_list(load_dataset("astune/mts_rl_dataset", streaming=True, split='train').skip(val_size).take(80).to_list())
    val_dataset = Dataset.from_list(load_dataset("astune/mts_rl_dataset", streaming=True, split='train').take(val_size).to_list())

    training_dataset = IterableMIDIDataset(train_dataset)
    val_dataset = IterableMIDIDataset(val_dataset)


    config = MTSGenConfig.mtsGen_150m()
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch59.pth'))
    model.to('cuda')

    trainer_config = TestRLConfig()
    trainer_config.save_dir = "../output/Model/rl_training_results"
    trainer_config.num_epochs = 500
    trainer_config.log_interval = 20
    trainer_config.eval_freq = 50
    trainer_config.checkpoint_freq = 10
    trainer_config.batch_size = 8
    trainer_config.sample_batch_factor = 10
    trainer_config.num_workers = 0
    trainer_config.collect_freq = 1

    trainer = RLTrainer(
        model=model,
        difficulty_system=GuitarSequenceAnalyzer(PresetConfigs.get_default()),
        similarity_system=MidiVersionComparator(),
        config=trainer_config,
    )
    #trainer.load_checkpoint("checkpoint_epoch_500.pt")
    trainer.logger.setLevel(logging.DEBUG)
    #trainer.load_checkpoint("")

    # 开始训练
    trainer.train(train_dataset=training_dataset, val_dataset = val_dataset)



