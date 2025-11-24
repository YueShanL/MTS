import json
import os
import warnings
from datetime import datetime

from peft import LoraConfig, get_peft_model, TaskType
from torch import ones_like
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenMelodyForConditionalGeneration

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 在导入 transformers 之前设置环境变量
import torch

print(f"PyTorch 版本: {torch.__version__}")

# 忽略 SDPA 警告
warnings.filterwarnings("ignore", message=".*SDPA requirements.*")


class SimpleMusicGenLoRATrainer:
    def __init__(self,
                 model_name="facebook/musicgen-melody",
                 output_dir="./lora_checkpoints",
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1):
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"设备: {self.device}")

        self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        self.audio_encoder = self.model.audio_encoder
        self.processor = AutoProcessor.from_pretrained(model_name)

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "linear1",
            "linear2",
            "lm_head"
        ]

        # 配置 LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )

        try:
            self.model = get_peft_model(self.model, self.lora_config)

            self.model.to(self.device)

            self.model.print_trainable_parameters()

        except ValueError as e:
            print(f"LoRA 应用失败: {e}")

    def create_dataloader(self, dataset, batch_size=1):
        """
        创建数据加载器
        """

        def collate_fn(batch):
            texts = [b['input_ids'] for b in batch]
            input_features = []
            labels = []
            sample_rate = 32000
            for i, b in enumerate(batch):
                input_features.append(b['input_features'])
                labels.append(b['labels'])
                if sample_rate == -1:
                    sample_rate = b['sample_rate']
                elif sample_rate != b['sample_rate']:
                    print(f'warming: sample_rate at idx {i} = {b["sample_rate"]} is differ from {sample_rate}')
                    continue

            # Process text
            inputs = self.processor(
                text=texts,
                is_split_into_words=True,
                audio=input_features,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )

            return {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs.get('attention_mask', ones_like(inputs['input_ids'])).to(
                    self.device),
                'input_features': inputs['input_features'].to(self.device),
                'labels': torch.stack(labels).to(self.device)
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            #shuffle=shuffle,
            collate_fn=collate_fn,
        )

        return dataloader

    def train_step(self, batch):
        """Single training step"""
        # Encode audio to get continuous representation
        try:
            labels = self.audio_encoder.encode(
                batch['labels'].to(torch.float16).to(self.device)
            ).audio_codes.squeeze(0).transpose(1, 2)

            # Forward pass with melody conditioning
            outputs = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                input_features=batch['input_features'].to(torch.float16).to(self.device),
                labels=labels.to(self.device)
            )

            loss = outputs.loss

        except Exception as e:
            print(e)
            return None

        return loss

    def train(self, dataloader, num_epochs=10, learning_rate=1e-4, save_every=1, save_best=True):
        """
        训练方法 - 添加检查点保存功能

        参数:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            save_every: 每隔多少轮保存一次检查点
            save_best: 是否保存最佳模型
        """
        # 检查是否使用了 LoRA
        is_lora = hasattr(self.model, 'peft_config') and self.model.peft_config is not None

        if is_lora:
            # LoRA 训练 - 只训练适配器参数
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
        else:
            # 全参数微调 - 训练所有参数
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate * 0.1  # 全参数微调使用更小的学习率
            )
            print("full params!")

        scaler = GradScaler()

        # 跟踪最佳损失
        best_loss = float('inf')

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx == 2001:
                    break
                if batch is None:
                    continue

                loss = self.train_step(batch)
                if loss is None:
                    continue

                print(f'loss = {loss.item()}')

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    mode = "LoRA" if is_lora else "全参数"
                    print(f"[{mode}] Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    torch.cuda.empty_cache()
                if batch_idx != 0 and batch_idx % 100 == 0:
                    self.save_checkpoint(epoch + 1, total_loss / num_batches, False)

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                mode = "LoRA" if is_lora else "全参数"
                print(f"[{mode}] Epoch {epoch + 1} 完成. 平均损失: {avg_loss:.4f}")

                # 保存检查点
                if (epoch + 1) % save_every == 0:
                    is_best = save_best and avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss

                    self.save_checkpoint(epoch + 1, avg_loss, is_best)

        # 训练完成后保存最终模型
        self.save_final_model()

    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        保存检查点
        """
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(checkpoint_dir)

        # 保存训练状态
        checkpoint_info = {
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "is_lora": hasattr(self.model, 'peft_config') and self.model.peft_config is not None
        }

        with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        print(f"检查点已保存: {checkpoint_dir}, 损失: {loss:.4f}")

        # 如果是最好模型，创建符号链接（在Windows上复制文件）
        if is_best:
            best_dir = os.path.join(self.output_dir, "best_model")
            if os.path.exists(best_dir):
                import shutil
                shutil.rmtree(best_dir)

            # 复制文件到best_model目录
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)
            print(f"最佳模型已保存: {best_dir}")

    def generate_music(self, text_prompts, max_length=1024, temperature=1.0):
        """
        使用训练后的LoRA生成音乐
        """
        self.model.eval()

        # 处理文本输入
        inputs = self.processor(
            text=text_prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 生成音频
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True
            )

        return audio_values

    def load_checkpoint(self, checkpoint_dir):
        """
        从检查点加载模型
        """
        try:
            from peft import PeftModel

            # 加载基础模型
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            )

            # 加载 LoRA 适配器
            self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            self.model.to(self.device)

            # 加载训练信息
            info_path = os.path.join(checkpoint_dir, "training_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    training_info = json.load(f)
                print(f"从检查点加载模型: {checkpoint_dir}")
                print(f"训练轮数: {training_info.get('epoch', '未知')}")
                print(f"损失: {training_info.get('loss', '未知')}")

            return True

        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False

    def save_final_model(self):
        """
        保存最终模型
        """
        final_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(final_dir)

        # 保存处理器
        self.processor.save_pretrained(final_dir)

        # 保存训练信息
        final_info = {
            "model_name": self.model_name,
            "saved_at": datetime.now().isoformat(),
            "is_lora": hasattr(self.model, 'peft_config') and self.model.peft_config is not None,
            "description": "MusicGen LoRA 微调模型"
        }

        with open(os.path.join(final_dir, "model_info.json"), "w") as f:
            json.dump(final_info, f, indent=2)

        print(f"最终模型已保存: {final_dir}")
