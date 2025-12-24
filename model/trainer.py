import math
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.loss import LossWrapper, AutoregressiveMultiTaskLoss
from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen, MixedTrainingForward


class TrainingConfig:
    """训练配置"""

    def __init__(self, **kwargs):
        # 默认配置
        defaults = {
            'num_epochs': 10,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'scheduler_type': 'linear',
            'min_tf_prob': 0.1,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'save_freq': 5,
            'eval_freq': 1,
        }

        # 更新用户配置
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)


class MixedTrainer:
    """简洁的混合训练器"""

    def __init__(self, model, loss_fn, config, scheduler_type='linear'):
        self.model = model
        self.loss_wrapper = LossWrapper(loss_fn)
        self.config = config
        self.scheduler = SamplingScheduler(scheduler_type)
        self.mixed_forward = MixedTrainingForward(model)

        # 训练统计
        self.stats = {'tf_used': 0, 'ar_used': 0}
        self._setup_optimizer()

    def _setup_optimizer(self):
        """设置优化器"""
        '''self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )'''
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # 从1e-5/1e-4/5e-4统一为更小的值
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 使用warmup策略
        self.scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-4,  # 峰值学习率
            epochs=self.config.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # 10%的时间用于warmup
            anneal_strategy='cos'
        )

    def train_step(self, batch, teacher_forcing_prob):
        """单步训练"""
        # 选择训练模式
        use_tf = random.random() < teacher_forcing_prob

        if use_tf:
            outputs = self.model(**batch, teacher_forcing=True)
            self.stats['tf_used'] += 1
        else:
            # 混合训练前向传播
            audio_features = self.model.encode_audio(batch['audio_input'])
            memory = self.model.fusion_encoder(audio_features)
            outputs = self.mixed_forward(memory, batch['target_notes'], teacher_forcing_prob)
            self.stats['ar_used'] += 1

        # 计算损失
        loss, details = self.loss_wrapper(outputs, batch['target_notes'], batch['audio_input'].device)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), details

    def train_epoch(self, dataloader, epoch, total_epochs):
        """训练一个epoch"""
        self.model.train()
        teacher_forcing_prob = self.scheduler.get_prob(epoch, total_epochs)
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{total_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            batch = self._move_to_device(batch)
            loss, _ = self.train_step(batch, teacher_forcing_prob)
            total_loss += loss

            progress_bar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = total_loss / len(dataloader)
        self.scheduler_lr.step(avg_loss)

        return avg_loss, teacher_forcing_prob

    def _move_to_device(self, batch):
        """移动批次数据到设备"""
        device = next(self.model.parameters()).device

        if 'audio_input' in batch:
            batch['audio_input'] = batch['audio_input'].to(device)

        for key in ['context_notes', 'target_notes']:
            if key in batch:
                for subkey in batch[key]:
                    batch[key][subkey] = batch[key][subkey].to(device)

        return batch

class MTSGenTrainer:

    def __init__(self, config=None, model = None):
        self.config = config or MTSGenConfig()
        self.model = model if model is not None else MTSGen(self.config)
        self._setup_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = AutoregressiveMultiTaskLoss(self.config, self.device)

    def _setup_optimizer(self):
        encoder_params = []
        decoder_params = []
        embedding_params = []

        for name, param in self.model.named_parameters():
            if 'audio_encoder' in name and self.config.freeze_encoder:
                param.requires_grad = False
            elif 'audio_encoder' in name:
                encoder_params.append(param)
            elif any(x in name for x in ['embedding', 'start_token']):
                embedding_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': 1e-5},
            {'params': decoder_params, 'lr': 1e-4},
            {'params': embedding_params, 'lr': 5e-4}
        ], weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

    def train(self, dataset, batch_size = 8, num_epoch = 20, output_path = ''):
        """完整的训练循环"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn, pin_memory=True,shuffle=True, num_workers=8)

        avg_loss = 0
        for epoch in range(num_epoch):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epoch}')

            for batch_idx, batch in enumerate(progress_bar):
                batch['audio_input'] = batch['audio_input'].to(device)
                for key in batch['context_notes'].keys():
                    batch['context_notes'][key] = batch['context_notes'][key].to(device)
                    batch['target_notes'][key] = batch['target_notes'][key].to(device)

                loss = self.train_step(batch)
                epoch_loss += loss


                progress_bar.set_postfix({'loss': f'{loss:.4f}'})

            # 计算平均损失
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}')

            # 更新学习率
            self.scheduler.step(avg_loss)

            # 可选：保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f'{output_path}/checkpoint_epoch_{epoch + 1}.pt')

        return avg_loss

    def train_step(self, batch):
        """训练步骤"""
        audio_input = batch['audio_input']
        context_notes = batch['context_notes']
        target_notes = batch['target_notes']

        # 前向传播
        outputs = self.model(
            audio_input=audio_input,
            context_notes=context_notes,
            target_notes=target_notes,
            do_sample=False,
            teacher_forcing=True
        )

        # 计算损失
        loss = self.loss_fn(outputs, target_notes, device=audio_input.device)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

class SamplingScheduler:
    """混合训练的概率调度器"""

    SCHEDULES = {
        'linear': lambda p, e, E: max(p.min_prob, 1.0 - e / E),
        'exponential': lambda p, e, E: max(p.min_prob, p.decay_rate ** e),
        'step': lambda p, e, E: {
            e < E * 0.3: 1.0,
            e < E * 0.6: 0.7,
            e < E * 0.8: 0.4
        }.get(True, p.min_prob),
        'inverse_sigmoid': lambda p, e, E: max(p.min_prob,
                                               p.k / (p.k + math.exp(e * p.k / (E - e + 1e-8))))
    }

    def __init__(self, schedule_type='linear', min_prob=0.1, decay_rate=0.95, k=5.0):
        self.schedule_type = schedule_type
        self.min_prob = min_prob
        self.decay_rate = decay_rate
        self.k = k

    def get_prob(self, epoch, total_epochs):
        """获取当前epoch的Teacher Forcing概率"""
        if self.schedule_type in self.SCHEDULES:
            return self.SCHEDULES[self.schedule_type](self, epoch, total_epochs)
        return 1.0  # 默认全Teacher Forcing

def collate_fn(batch):
        if not batch:
            return {}

        # 预先确定键，避免动态扩展字典（小幅优化）
        first_sample = batch[0]
        audio_list = []

        # 预先分配列表，假设键固定为这三个
        context_duration = []
        context_fret = []
        context_technique = []
        target_duration = []
        target_fret = []
        target_technique = []

        for sample in batch:
            # 1. 音频数据
            audio_list.append(Tensor(sample['audio_input']))

            # 2. 上下文数据 - 直接提取，假设结构固定
            context_notes = sample['context_notes']
            context_duration.append(Tensor(context_notes['duration']))
            context_fret.append(Tensor(context_notes['fret']))
            context_technique.append(Tensor(context_notes['technique']))

            # 3. 目标数据
            target_notes = sample['target_notes']
            target_duration.append(Tensor(target_notes['duration']))
            target_fret.append(Tensor(target_notes['fret']))
            target_technique.append(Tensor(target_notes['technique']))

        # 使用torch.stack一次性堆叠，减少碎片化操作
        batched = {
            'audio_input': torch.stack(audio_list).unsqueeze(1),  # [B, 1, T]
            'context_notes': {
                'duration': torch.stack(context_duration).to(torch.int64),
                'fret': torch.stack(context_fret).to(torch.int64),
                'technique': torch.stack(context_technique).to(torch.int64)
            },
            'target_notes': {
                'duration': torch.stack(target_duration).to(torch.int64),
                'fret': torch.stack(target_fret).to(torch.int64),
                'technique': torch.stack(target_technique).to(torch.int64)
            }
        }

        return batched

def train_mixed_model(model, train_dataset, val_dataset=None,
                      num_epochs=10, batch_size=8, scheduler_type='linear', output_path='.'):
    """完整的混合训练循环"""

    # 初始化组件
    config = model.config
    loss_fn = AutoregressiveMultiTaskLoss(config, use_focal=True)
    trainer = MixedTrainer(model, loss_fn, config, scheduler_type)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=collate_fn, shuffle=True, num_workers=4)

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                collate_fn=collate_fn, num_workers=2)

    # 训练循环
    training_log = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        train_loss, tf_prob = trainer.train_epoch(train_loader, epoch, num_epochs)

        # 验证
        val_loss = None
        if val_dataset:
            val_loss = evaluate_model(model, loss_fn, val_loader)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{output_path}/best_model_epoch{epoch}.pth')

        # 记录日志
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tf_prob': tf_prob,
            'tf_used': trainer.stats['tf_used'],
            'ar_used': trainer.stats['ar_used']
        }
        training_log.append(log_entry)

        # 打印进度
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss if val_loss else "N/A":.4f}, '
              f'TF Prob: {tf_prob:.2f}, '
              f'TF/AR: {trainer.stats["tf_used"]}/{trainer.stats["ar_used"]}')

    return training_log


def evaluate_model(model, loss_fn, dataloader):
    """评估模型性能"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            device = next(model.parameters()).device
            batch['audio_input'] = batch['audio_input'].to(device)
            for key in batch['target_notes']:
                batch['target_notes'][key] = batch['target_notes'][key].to(device)

            # 自回归生成（模拟推理）
            _, logits = model(
                audio_input=batch['audio_input'],
                context_notes=None,
                target_notes=None,
                teacher_forcing=False,
                generate_length=64,
                return_logits=True
            )

            # 计算损失
            loss, _ = loss_fn(logits, batch['target_notes'])
            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)