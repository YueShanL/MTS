#!/usr/bin/env python3
"""
Complete Music Generation Script
Ready-to-use inference script
"""

import torch
import torchaudio
import argparse
import os
from training.compressed_musicgen_complete import CompressedMusicGen

def generate_music(model_path, prompts, compression_factors, output_dir):
    """Generate music from prompts"""

    # Load model
    print(f"Loading model from {model_path}...")
    model = CompressedMusicGen.load_model(model_path)
    model.eval()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Generate music
    for i, (prompt, compression) in enumerate(zip(prompts, compression_factors)):
        print(f"\nGenerating {i+1}/{len(prompts)}: '{prompt}' at {compression}x speed")

        try:
            # Generate
            with torch.no_grad():
                audio = model.generate(prompt, compression_factor=compression)

            # Save
            filename = f"generated_{i+1:03d}_{compression}x.wav"
            filepath = os.path.join(output_dir, filename)

            torchaudio.save(filepath, audio.cpu().unsqueeze(0), model.config.sample_rate)

            duration = audio.shape[0] / model.config.sample_rate
            print(f"✅ Saved: {filepath} (duration: {duration:.1f}s)")

        except Exception as e:
            print(f"❌ Error generating '{prompt}': {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate music with CompressedMusicGen')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--output_dir', default='generated_music', help='Output directory')
    parser.add_argument('--prompt', action='append', help='Music prompts (can use multiple times)')
    parser.add_argument('--compression', type=float, action='append', help='Compression factors')
    parser.add_argument('--batch_generate', action='store_true', help='Generate predefined batch')

    args = parser.parse_args()

    if args.batch_generate:
        # Predefined batch generation
        prompts = [
            "upbeat electronic music with synthesizer",
            "acoustic guitar melody peaceful",
            "energetic rock music with drums",
            "classical piano instrumental",
            "ambient electronic soundscape",
            "jazz improvisation with saxophone",
            "folk acoustic song gentle",
            "electronic dance music fast"
        ]
        compression_factors = [2.0, 1.5, 2.5, 1.2, 2.0, 1.8, 1.5, 3.0]
    else:
        # Use provided prompts
        if not args.prompt:
            prompts = ["electronic music"]
        else:
            prompts = args.prompt

        if not args.compression:
            compression_factors = [2.0] * len(prompts)
        else:
            compression_factors = args.compression
            # Extend or truncate to match prompts
            while len(compression_factors) < len(prompts):
                compression_factors.append(2.0)
            compression_factors = compression_factors[:len(prompts)]

    print(f"🎵 CompressedMusicGen - Music Generation")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Prompts: {len(prompts)}")

    generate_music(args.model_path, prompts, compression_factors, args.output_dir)

    print(f"\n🎉 Generation completed! Check {args.output_dir}")

if __name__ == "__main__":
    main()
