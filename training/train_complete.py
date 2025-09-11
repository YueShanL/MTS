#!/usr/bin/env python3
"""
Complete Training Script for CompressedMusicGen
Ready-to-run training implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from compressed_musicgen_complete import CompressedMusicGen, ModelConfig

class MusicDataset(Dataset):
    """Simple music dataset for training"""

    def __init__(self, data_dir, config, split='train'):
        self.data_dir = data_dir
        self.config = config

        # Load data list
        data_file = os.path.join(data_dir, f'{split}_data.json')
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.data_list = json.load(f)
        else:
            # Create synthetic data for demo
            self.data_list = self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Create synthetic training data"""
        descriptions = [
            "electronic music with synthesizer",
            "acoustic guitar melody",
            "piano instrumental",
            "rock music with drums",
            "ambient soundscape",
            "jazz improvisation",
            "classical orchestral",
            "pop music upbeat",
            "folk acoustic song",
            "electronic dance music"
        ]

        data_list = []
        for i in range(100):  # 100 synthetic samples
            data_list.append({
                'description': descriptions[i % len(descriptions)],
                'audio_length': np.random.randint(10, 20) * self.config.sample_rate,
                'index': i
            })

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Generate synthetic audio (for demo)
        audio_length = item.get('audio_length', self.config.sample_rate * 15)
        audio = torch.randn(audio_length) * 0.1

        # Pad or truncate to fixed length
        target_length = self.config.max_audio_length
        if audio.size(0) > target_length:
            start_idx = np.random.randint(0, audio.size(0) - target_length)
            audio = audio[start_idx:start_idx + target_length]
        elif audio.size(0) < target_length:
            padding = target_length - audio.size(0)
            audio = torch.cat([audio, torch.zeros(padding)])

        # Simple text tokenization
        description = item['description']
        words = description.lower().split()
        tokens = [hash(word) % self.config.vocab_size for word in words]
        tokens = tokens[:50] + [0] * max(0, 50 - len(tokens))

        return {
            'text_ids': torch.tensor(tokens, dtype=torch.long),
            'text_mask': torch.ones(50, dtype=torch.long),
            'audio': audio,
            'compression_factor': np.random.choice([1.5, 2.0, 2.5])
        }

class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move to device
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)
            audio = batch['audio'].to(self.device)
            compression_factor = batch['compression_factor'][0].item()

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text_ids, audio, text_mask, compression_factor)
                    loss = outputs['loss']

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(text_ids, audio, text_mask, compression_factor)
                loss = outputs['loss']

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                text_ids = batch['text_ids'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)
                audio = batch['audio'].to(self.device)
                compression_factor = batch['compression_factor'][0].item()

                outputs = self.model(text_ids, audio, text_mask, compression_factor)
                loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device to use')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup model and data
    config = ModelConfig()
    model = CompressedMusicGen(config)

    train_dataset = MusicDataset(args.data_dir, config, 'train')
    val_dataset = MusicDataset(args.data_dir, config, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = trainer.train_epoch(epoch)

        # Validate
        val_loss = trainer.validate()

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(os.path.join(args.output_dir, 'best_model'))

        # Save checkpoint
        trainer.save_checkpoint(epoch, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

        # Test generation
        if epoch % 5 == 0:
            print("\nTesting generation...")
            test_audio = model.generate("electronic music", compression_factor=2.0)
            torchaudio.save(
                os.path.join(args.output_dir, f'test_epoch_{epoch}.wav'),
                test_audio.cpu().unsqueeze(0),
                config.sample_rate
            )
            print(f"Test audio saved: test_epoch_{epoch}.wav")

    print("\n🎉 Training completed!")
    print(f"Best model saved in: {os.path.join(args.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()
