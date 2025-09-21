import os.path

from data.loader import load_piast_dataset

debug = 0
continue_on_exception = 1
linux = 1
if __name__ == '__main__':
    dataset = load_piast_dataset(repo_path="data/dataset/PIAST", download_if_empty=True) if linux \
        else load_piast_dataset(repo_path="../data/dataset/PIAST", download_if_empty=True)
    output_path = "output/Lora/training/" if linux else "../output/Lora/training/"
    print(f'get dataset {dataset}')

    if debug:
        print(f'debug mode!')

        subset = dataset['piast-yt']
        for s in subset:
            if 'rog6LSvp8MY' in s['midi_path']:
                source_path = s['midi_path']
                tag = str(s['text'])

        # source_path = subset[0]['midi_path']
        # tag = str(subset[0]['text'])
        # print(subset[:10])

        from data.generator import generate

        generate(source_path,
                 tag,
                 output_path,
                 repeating_limit=1,
                 fix_style="Rock",
                 time_limit=30,
                 split_audio=10)

    else:
        subset = dataset['piast-yt']
        source_path = subset['midi_path']
        tag = subset['text']

        from data.generator import generate

        for path, text in zip(source_path, tag):
            try:
                generate(path, text, output_path, time_limit=1200, split_audio=10)
            except Exception as e:
                print(f'failed on generating {os.path.basename(path)} because: {e}')
                if continue_on_exception: continue
