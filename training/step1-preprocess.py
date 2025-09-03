from openunmix import predict
import torchaudio
from audiocraft.data.audio import audio_write

testSource = "./asset/"
filename = 'songTest.mp3'
'umxhq'

melody, sr = torchaudio.load(testSource + filename)

def preprocess(dataset, ):
    separate_song()
    vocal_to_note()
    combine_melody()
    speedShift()
    slice()
    return


def separate_song(audio, sr, export: str = None, target=None, modelName="umxl"):
    separated = predict.separate(audio[None], rate=sr, model_str_or_path=modelName, targets=target)
    if export is not None:
        for name, tensor in zip(separated.keys(), separated.values()):
            # Will save under {modelName}.{export}.{name}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{modelName}.{export}.{name}', tensor[0].cpu(), sr, strategy="loudness",
                        loudness_compressor=True)
    return separated


# TODO: implements below functions. target: highlight mainstream melody in vocal music
def vocal_to_note(audio, export: str = None, ):
    return


def combine_melody():
    return


def slice():
    return


def speedShift():
    return
