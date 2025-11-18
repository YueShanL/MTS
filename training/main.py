from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from data.loader import load_piast_dataset
from dataset_builder import load_audio_dataset, process_style_transfer_dataset
from training.dataset import MusicGenMelodyDataset
from training.lora import SimpleMusicGenLoRATrainer
from transformers import AutoProcessor

from utils.debug import check_lora_dataset

debug = False


def main():
    # 1. 初始化训练器
    trainer = SimpleMusicGenLoRATrainer(
        model_name='facebook/musicgen-melody',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # piast = load_piast_dataset('../data/dataset/PIAST/')

    dataset = DatasetDict.load_from_disk("../output/Lora/dataset")

    if debug: check_lora_dataset(dataset)

    dataset = MusicGenMelodyDataset(
        dataset['train'],
        processor=AutoProcessor.from_pretrained('facebook/musicgen-melody')
    )

    train_dataloader = trainer.create_dataloader(dataset)
    print(f'load dataset of size {train_dataloader.dataset.__len__()}')

    # 4. 开始训练
    trainer.train(
        dataloader=train_dataloader,
        num_epochs=10,
        learning_rate=1e-4,
    )

    # 5. 测试生成
    print("测试音乐生成...")
    prompts = ["happy electronic music", "piano cover"]
    generated_audio = trainer.generate_music(prompts)

    # 保存生成的音频
    for i, audio in enumerate(generated_audio):
        import soundfile as sf
        sf.write(f"generated_{i}.wav", audio.cpu().numpy(), 32000)


if __name__ == "__main__":
    main()
