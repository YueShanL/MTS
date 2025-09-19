import librosa
import pretty_midi
import numpy as np
import torch
import os
from pathlib import Path

from audiocraft.data.audio import audio_write


def midi_to_audio_tensor(midi_path, sr=44100, duration=None, return_numpy=False,
                        debug=False, debug_dir="debug_output", save_audio=False,
                        visualize=False, normalize=True, soundfont_path=None):
    """
    将 MIDI 文件转换为音频形式的张量，并提供调试选项

    参数:
        midi_path (str): MIDI 文件的路径
        sr (int): 采样率，默认为 22050
        duration (float): 音频持续时间（秒），如果为None则使用MIDI文件的持续时间
        return_numpy (bool): 如果为True，返回numpy数组；否则返回PyTorch张量
        debug (bool): 是否启用调试模式
        debug_dir (str): 调试输出目录
        save_audio (bool): 是否保存音频文件
        visualize (bool): 是否生成可视化图表
        normalize (bool): 是否归一化音频信号

    返回:
        audio_tensor (torch.Tensor或numpy.ndarray): 形状为 (samples,) 的音频张量
        sr (int): 采样率
    """
    try:
        # 创建调试输出目录
        if debug and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # 加载MIDI文件
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # 调试信息：打印MIDI文件基本信息
        if debug:
            print(f"MIDI文件信息: {midi_path}")
            print(f"  持续时间: {midi_data.get_end_time():.2f} 秒")
            print(f"  乐器数量: {len(midi_data.instruments)}")
            for i, instrument in enumerate(midi_data.instruments):
                print(f"    乐器 {i}: 程序={instrument.program}, 是否为鼓={instrument.is_drum}, 音符数量={len(instrument.notes)}")

        # 获取MIDI文件的持续时间
        midi_duration = midi_data.get_end_time()

        # 如果未指定持续时间，使用MIDI文件的完整持续时间
        if duration is None:
            duration = midi_duration

        # 使用SoundFont合成音频（如果提供了SoundFont路径）
        if soundfont_path and os.path.exists(soundfont_path):
            if debug:
                print(f"使用SoundFont合成音频: {soundfont_path}")

            try:
                # 使用fluidsynth合成音频
                import fluidsynth
                sf = fluidsynth.Synth()
                sf.start()

                # 合成音频
                audio_signal = midi_data.fluidsynth(fs=sr, sf2_path=soundfont_path)

            except ImportError:
                print("警告: 未安装fluidsynth，使用内置合成器")
                print("请运行: pip install pyfluidsynth")
                audio_signal = midi_data.synthesize(fs=sr)
        else:
            if debug and soundfont_path:
                print(f"警告: SoundFont文件不存在: {soundfont_path}，使用内置合成器")
            # 使用内置合成器
            audio_signal = midi_data.synthesize(fs=sr)

        # 确保音频信号长度正确
        expected_length = int(sr * duration)
        if len(audio_signal) < expected_length:
            # 如果音频信号太短，进行填充
            audio_signal = np.pad(audio_signal, (0, expected_length - len(audio_signal)))
        elif len(audio_signal) > expected_length:
            # 如果音频信号太长，进行截断
            audio_signal = audio_signal[:expected_length]

        # 归一化音频信号
        if normalize and np.max(np.abs(audio_signal)) > 0:
            audio_signal = audio_signal / np.max(np.abs(audio_signal))

        # 保存音频文件（用于调试）
        if debug and save_audio:
            try:
                import soundfile as sf
                audio_filename = os.path.join(debug_dir, f"{Path(midi_path).stem}_audio.wav")
                sf.write(audio_filename, audio_signal, sr)
                print(f"音频文件已保存: {audio_filename}")
            except ImportError:
                print("警告: 需要安装soundfile库才能保存音频文件。请运行: pip install soundfile")

        # 生成可视化图表（用于调试）
        if debug and visualize:
            try:
                import matplotlib.pyplot as plt

                # 创建波形图
                plt.figure(figsize=(12, 4))
                plt.plot(np.arange(len(audio_signal)) / sr, audio_signal)
                plt.title(f"音频波形: {Path(midi_path).stem}")
                plt.xlabel("时间 (秒)")
                plt.ylabel("振幅")
                plt.tight_layout()
                waveform_path = os.path.join(debug_dir, f"{Path(midi_path).stem}_waveform.png")
                plt.savefig(waveform_path)
                plt.close()
                print(f"波形图已保存: {waveform_path}")

                # 创建频谱图
                plt.figure(figsize=(12, 6))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal)), ref=np.max)
                librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"频谱图: {Path(midi_path).stem}")
                plt.tight_layout()
                spectrogram_path = os.path.join(debug_dir, f"{Path(midi_path).stem}_spectrogram.png")
                plt.savefig(spectrogram_path)
                plt.close()
                print(f"频谱图已保存: {spectrogram_path}")

            except ImportError:
                print("警告: 需要安装matplotlib和librosa库才能生成可视化图表。")
                print("请运行: pip install matplotlib librosa")

        # 转换为张量
        if return_numpy:
            return audio_signal, sr
        else:
            return torch.from_numpy(audio_signal).float(), sr

    except Exception as e:
        print(f"处理MIDI文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, sr


if __name__ == "__main__":
    file_name = "7iPSSj62CUw.mid"
    soundfont_path = "..\\asset\\GeneralUser-GS.sf2"
    # 将MIDI文件转换为音频张量
    # 启用调试模式
    audio_tensor, sr = midi_to_audio_tensor(
        file_name,
        debug=True,
        save_audio=True,
        visualize=True,
        soundfont_path=soundfont_path
    )

    if audio_tensor is not None:
        print(f"音频张量形状: {audio_tensor.shape}")
        print(f"采样率: {sr}")
        print(f"最大值: {audio_tensor.max().item()}, 最小值: {audio_tensor.min().item()}")
