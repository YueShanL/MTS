import os.path

from datasets import DatasetDict
from transformers import AutoProcessor

from data.loader import load_piast_dataset
from training.dataset import MusicGenMelodyDataset
from training.dataset_builder import load_audio_dataset, process_style_transfer_dataset
from training.lora import SimpleMusicGenLoRATrainer

debug = 0
continue_on_exception = 1
linux = 1
if __name__ == '__main__':
    dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset"
    generated_audio_dir = "output/Lora/training" if linux else "../output/Lora/training"
    output_path = "output/Lora/adaptor/checkpoints/" if linux else "../output/Lora/adaptor/checkpoints/"

    generating_dataset = False

    try:
        dataset = DatasetDict.load_from_disk(dataset_path)
    except Exception as e:
        print(f'failed to load from {dataset_path}, trying to generate dataset')
        generating_dataset = True

    if generating_dataset:

        piast = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
            else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)

        data = load_audio_dataset(
            piast_data=piast['piast-yt'],
            generated_audio_dir=generated_audio_dir
        )

        data = process_style_transfer_dataset(data)

        print(data.info)
        DatasetDict({"train": data}).save_to_disk(dataset_path)
    else:
        data = dataset['train']

    trainer = SimpleMusicGenLoRATrainer(
        model_name='facebook/musicgen-melody',
        output_dir=output_path,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    dataset = MusicGenMelodyDataset(
        data,
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
