import random

from datasets import Features, Value, Sequence, DatasetDict
import soundfile as sf

debug = 0
continue_on_exception = 1
linux = 1
if __name__ == '__main__':
    dataset_path = "output/Lora/dataset/splited" if linux else "../output/Lora/dataset/splited"
    generated_audio_dir = "output/Lora/training" if linux else "../output/Lora/training"
    output_path = "Test/output/dataset_samples/lora/" if linux else "output/dataset_samples/lora/"

    generating_dataset = False

    features = Features({
        'text': Value('string'),
        'input_audio_values': Sequence(Value('float32')),
        'target_audio_values': Sequence(Value('float32'))
    })

    dataset = DatasetDict.load_from_disk(dataset_path)['train']

    sf.write(f"{output_path}out.wav",  dataset[random.randint(0, len(dataset) - 1)]['target_audio_values'].squezze(), 32000)