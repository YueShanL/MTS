from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH


def transcribe(audio_path, output_midi_path):
    predict_and_save(
        audio_path,
        output_midi_path,
        True,
        True,
        False,
        False,
        ICASSP_2022_MODEL_PATH
    )


if __name__ == '__main__':
    transcribe(["debug_output/7iPSSj62CUw_audio.wav"], "debug_output")