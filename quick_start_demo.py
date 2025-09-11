#!/usr/bin/env python3
"""
Quick Start Demo - Test the complete system
"""

import torch
import torchaudio
import os
from training.compressed_musicgen_complete import CompressedMusicGen, ModelConfig

def quick_demo():
    print("🎵 CompressedMusicGen Quick Demo")
    print("=" * 50)

    # 1. Create model
    print("\n1. Creating model...")
    model = CompressedMusicGen()
    print(f"✅ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 2. Test generation (untrained model)
    print("\n2. Testing generation...")

    prompts = [
        ("electronic music", 2.0),
        ("piano melody", 1.5),
        ("rock music", 2.5)
    ]

    os.makedirs("demo_output", exist_ok=True)

    for i, (prompt, speed) in enumerate(prompts):
        print(f"   Generating: '{prompt}' at {speed}x speed...")

        try:
            audio = model.generate(prompt, compression_factor=speed)

            filename = f"demo_output/demo_{i+1}_{speed}x.wav"
            torchaudio.save(filename, audio.unsqueeze(0), model.config.sample_rate)

            duration = audio.shape[0] / model.config.sample_rate
            print(f"   ✅ Saved: {filename} ({duration:.1f}s)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 3. Test training step
    print("\n3. Testing training step...")

    try:
        # Synthetic training data
        batch_size = 2
        text_ids = torch.randint(0, 8000, (batch_size, 50))
        text_mask = torch.ones(batch_size, 50)
        audio = torch.randn(batch_size, model.config.max_audio_length)

        model.train()
        outputs = model(text_ids, audio, text_mask, compression_factor=2.0)

        print(f"   ✅ Training step successful (loss: {outputs['loss'].item():.4f})")

    except Exception as e:
        print(f"   ❌ Training error: {e}")

    # 4. Test save/load
    print("\n4. Testing save/load...")

    try:
        # Save
        model.save_model("demo_output/test_model")
        print("   ✅ Model saved")

        # Load
        loaded_model = CompressedMusicGen.load_model("demo_output/test_model")
        print("   ✅ Model loaded")

    except Exception as e:
        print(f"   ❌ Save/load error: {e}")

    print("\n🎉 Quick demo completed!")
    print("\nGenerated files in demo_output/:")
    if os.path.exists("demo_output"):
        for f in os.listdir("demo_output"):
            print(f"  - {f}")

    print("\n📋 Next steps:")
    print("1. python train_complete.py --epochs 20")
    print("2. python generate_music.py --model_path outputs/best_model --batch_generate")

if __name__ == "__main__":
    quick_demo()
