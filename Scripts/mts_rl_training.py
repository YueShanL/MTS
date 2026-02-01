import torch

from data.loader import load_piast_dataset
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.rl.dataset import MIDISegmentDataset
from model.rl.mid_comparitor import MidiVersionComparator
from model.rl.simulator import GuitarSequenceAnalyzer, PresetConfigs
from model.rl.trainer import RLTrainer, RLConfig, TestRLConfig

debug = 0
continue_on_exception = 1
linux = 0

if __name__ == "__main__":
    dataset = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

    size = len(dataset["piast-yt"])

    train_val_split = dataset["piast-yt"].select(range(size)).train_test_split(
        test_size=0.1,
        seed=42,
        shuffle=True
    )

    training_dataset = MIDISegmentDataset(
        midi_dataset=train_val_split['train'],
        batch=16,
        length_seconds=8,
        sample_rate=24000,
        overlap_ratio=0.3,
        shuffle=True,
        infinite=True,
        min_notes_per_segment=3,
        random_start=True
    )
    eval_dataset = MIDISegmentDataset(
        midi_dataset=train_val_split['test'],
        batch=16,
        length_seconds=8,
        sample_rate=24000,
        overlap_ratio=0.3,
        shuffle=True,
        infinite=True,
        min_notes_per_segment=3,
        random_start=True
    )

    stats = training_dataset.get_stats()
    print("数据集统计:", stats)

    config = MTSGenConfig.mtsGen_300m_depth()
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch19.pth'))
    model.to('cuda')

    trainer_config = TestRLConfig()
    trainer_config.save_dir = "../output/Model/rl_training_results"

    trainer = RLTrainer(
        model=model,
        difficulty_system=GuitarSequenceAnalyzer(PresetConfigs.get_default()),
        similarity_system=MidiVersionComparator(),
        config=trainer_config,
    )
    #trainer.load_checkpoint("")

    # 开始训练
    trainer.train(train_dataset=training_dataset, val_dataset = eval_dataset)



