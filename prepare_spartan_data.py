#!/usr/bin/env python3
"""
为 Spartan HPC 准备 CompressedMusicGen 训练数据
"""

import os
import torch
import torchaudio
import json
import numpy as np
from pathlib import Path

def prepare_synthetic_dataset(output_dir="data/spartan_training", num_samples=1000):
    """准备合成训练数据集"""
    
    print(f"🎵 准备 Spartan 训练数据集")
    print(f"输出目录: {output_dir}")
    print(f"样本数量: {num_samples}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 音乐风格描述
    descriptions = [
        "upbeat electronic music with synthesizer",
        "acoustic guitar melody peaceful",
        "energetic rock music with drums", 
        "classical piano instrumental",
        "ambient electronic soundscape",
        "jazz improvisation with saxophone",
        "folk acoustic song gentle",
        "pop music catchy melody",
        "blues guitar slow tempo",
        "orchestral classical music"
    ]
    
    # 生成训练数据
    train_data = []
    val_data = []
    
    for i in range(num_samples):
        # 选择描述
        desc = descriptions[i % len(descriptions)]
        
        # 生成合成音频 (15秒 @ 22kHz)
        duration = 15
        sample_rate = 22050
        audio_length = duration * sample_rate
        
        # 创建合成音频信号
        t = torch.linspace(0, duration, audio_length)
        
        # 基础频率和谐波
        base_freq = np.random.uniform(200, 800)  # 基础频率
        audio = torch.sin(2 * np.pi * base_freq * t)
        
        # 添加谐波
        for harmonic in range(2, 5):
            amplitude = 1.0 / harmonic
            audio += amplitude * torch.sin(2 * np.pi * base_freq * harmonic * t)
        
        # 添加包络
        envelope = torch.exp(-t / 3.0)  # 衰减包络
        audio = audio * envelope
        
        # 归一化
        audio = audio / torch.max(torch.abs(audio))
        audio = audio * 0.8  # 防止削波
        
        # 添加少量噪声
        noise = torch.randn_like(audio) * 0.01
        audio = audio + noise
        
        # 准备数据项
        data_item = {
            'audio': audio.tolist(),
            'description': desc,
            'sample_rate': sample_rate,
            'duration': duration,
            'compression_factors': [1.5, 2.0, 2.5]  # 支持的压缩倍率
        }
        
        # 8:2 训练验证分割
        if i < num_samples * 0.8:
            train_data.append(data_item)
        else:
            val_data.append(data_item)
        
        if (i + 1) % 100 == 0:
            print(f"生成进度: {i+1}/{num_samples}")
    
    # 保存数据集
    with open(f"{output_dir}/train_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(f"{output_dir}/val_data.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ 数据集准备完成!")
    print(f"训练样本: {len(train_data)}")
    print(f"验证样本: {len(val_data)}")
    
    return output_dir

if __name__ == "__main__":
    prepare_synthetic_dataset()
