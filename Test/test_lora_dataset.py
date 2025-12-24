import random

from datasets import Features, Value, Sequence, Dataset, DatasetDict
import soundfile as sf

debug = 0
continue_on_exception = 1
linux = 0
if __name__ == '__main__':
    dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset"
    generated_audio_dir = "output/Lora/training" if linux else "../output/Lora/training"
    output_path = "Test/output/dataset_samples/lora/" if linux else "output/dataset_samples/lora/"

    generating_dataset = False

    features = Features({
        'text': Value('string'),
        'input_audio_values': Sequence(Value('float32')),
        'target_audio_values': Sequence(Value('float32'))
    })

    dataset = DatasetDict.load_from_disk(dataset_path)['train']
    idx = random.randint(0, len(dataset) - 1)
    sf.write(f"{output_path}out.wav",  dataset[idx]['target_audio_values'], 32000)
    sf.write(f"{output_path}in.wav",  dataset[idx]['input_audio_values'], 32000)
