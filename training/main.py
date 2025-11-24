import datasets
from datasets import Features, Value, Sequence
from transformers import AutoProcessor

from training.dataset import create_musicgen_dataset
from training.lora import SimpleMusicGenLoRATrainer
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

    features = Features({
        'text': Value('string'),
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int8')),
        'input_audio_values': Sequence(Value('float32')),
        'target_audio_values': Sequence(Value('float32'))
    })

    dataset = datasets.load_dataset(path="../output/Lora/dataset", features=features, split='train', streaming=True)

    if debug: check_lora_dataset(dataset)

    dataset = create_musicgen_dataset(
        dataset,
        processor=AutoProcessor.from_pretrained('facebook/musicgen-melody')
    )

    train_dataloader = trainer.create_dataloader(dataset)

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
