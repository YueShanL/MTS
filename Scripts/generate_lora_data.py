from data.loader import load_piast_dataset

if __name__ == '__main__':
    dataset = load_piast_dataset(repo_path="../data/dataset/PIAST")
    output_path = "../output/Lora/training/"

    print(f'get dataset {dataset}')

    subset = dataset['piast-yt']
    source_path = subset[1]['midi_path']
    tag = str(subset[1]['text'])
    print(subset[:10])

    from data.generator import generate

    generate(source_path, tag, output_path, time_limit=180)
