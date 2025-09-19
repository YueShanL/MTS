from data.loader import load_piast_dataset
debug = 0

if __name__ == '__main__':
    dataset = load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)
    output_path = "../output/Lora/training/"
    print(f'get dataset {dataset}')

    if debug:
        print(f'debug mode!')

        subset = dataset['piast-yt']
        source_path = subset[1]['midi_path']
        tag = str(subset[1]['text'])
        print(subset[:10])

        from data.generator import generate

        generate(source_path, tag, output_path, time_limit=180)

    else:
        subset = dataset['piast-yt']
        source_path = subset['midi_path']
        tag = subset['text']

        from data.generator import generate

        for path, text in zip(source_path, tag):
            generate(path, text, output_path)