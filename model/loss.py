import torch
from torch import nn

from model.focal_loss import FocalLoss
import torch.nn.functional as F


class LossWrapper:
    """损失函数包装器，支持混合训练输出"""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, predictions, targets, device='cuda'):
        """计算损失，自动处理混合训练输出格式"""
        # 提取logits（如果是混合训练输出）
        logits = predictions.get('logits', predictions)

        # 计算损失
        total_loss, loss_details = self.loss_fn(logits, targets)

        # 如果是混合训练，返回预测结果用于监控
        if 'predictions' in predictions:
            loss_details['predictions'] = predictions['predictions']

        return total_loss, loss_details


class AutoregressiveMultiTaskLoss(nn.Module):
    """自回归多任务损失函数（每根弦独立），集成Focal Loss"""

    def __init__(self, config, use_focal=False,
                 gamma=2.0, alpha_scale=2.0, device = 'cuda'):
        """
        Args:
            config: 模型配置
            weights: 各任务权重
            use_focal: 是否使用Focal Loss
            gamma: Focal Loss的gamma参数
            alpha_scale: 类别平衡权重缩放因子
        """
        super().__init__()
        self.config = config
        self.use_focal = use_focal
        self.gamma = gamma
        self.device = device

        # 设置类别权重和Focal Loss参数
        self._setup_class_weights(alpha_scale)

        # 初始化Focal Loss实例
        if use_focal:
            self._init_focal_losses()

    def _setup_class_weights(self, alpha_scale):
        """设置吉他任务特定的权重 - 基于真实数据分布优化"""

        # ============ Duration权重 ============
        # 你的数据：0:48.7%, 1:1.1%, 2:1.3%, 3:2.5%, 4:3.0%, 5:7.2%, 6:8.2%, 7:14.8%, 9:13.1%
        # 策略：大幅惩罚休止符(0)，鼓励其他持续时间
        self.duration_alpha = torch.ones(self.config.num_durations).to(self.device)

        # 设置具体权重（根据频率反比调整）
        dur_weights = {
            0: 0.2,   # 休止符：48.7% -> 权重0.05 (强烈惩罚)
            1: 3.5,    # 非常罕见但重要的时值：1.1% -> 权重3.5 (高奖励)
            2: 3.0,    # 1.3% -> 权重3.0
            3: 2.0,    # 2.5% -> 权重2.0
            4: 1.5,    # 3.0% -> 权重1.5
            5: 0.8,    # 7.2% -> 权重0.8
            6: 0.7,    # 8.2% -> 权重0.7
            7: 0.5,    # 14.8% -> 权重0.5
            9: 0.6,    # 13.1% -> 权重0.6
        }

        for idx, weight in dur_weights.items():
            if idx < self.config.num_durations:
                self.duration_alpha[idx] = weight

        print(f"Duration权重设置: 休止符(0)->{dur_weights[0]:.2f}, 1->{dur_weights[1]:.2f}, ...")

        # ============ Fret权重 ============
        # 你的数据：fret 25(不演奏)占77.4%，真实品位(0-24)共占22.6%
        # 策略：极度惩罚"不演奏"，大幅提升真实品位权重
        self.fret_alpha = torch.ones(self.config.max_fret + 2).to(self.device)

        # 不演奏标记(fret 25)极度惩罚
        self.fret_alpha[-1] = 0.2  # 77.4% -> 权重0.08

        # 真实品位权重策略：
        # 1. 常用低把位(0-12)：中等奖励（频率相对较高）
        # 2. 高把位(13-24)：高额奖励（更稀有）

        # 常用低把位(0-12)：权重1.5-2.5
        for i in range(0, 13):
            if i == 0:  # 空弦
                self.fret_alpha[i] = 2.0
            elif i in [5, 7, 12]:  # 常用品位
                self.fret_alpha[i] = 1.5
            else:
                self.fret_alpha[i] = 2.0

        # 高把位(13-24)：更高的奖励（更稀有）
        for i in range(13, 25):
            if i == 23:  # 你的数据显示23品有3.9%，相对较高
                self.fret_alpha[i] = 1.0
            else:
                self.fret_alpha[i] = 3.0  # 其他高把位给高奖励

        # 特殊：根据你的实际数据调整某些品位的权重
        # 例如，你的数据显示fret 23有3.9%，已经做了处理

        print(f"Fret权重设置: 不演奏(25)->{self.fret_alpha[-1]:.2f}, 空弦(0)->{self.fret_alpha[0]:.2f}, 高把位(20)->{self.fret_alpha[20]:.2f}")

        # ============ Technique权重 ============
        # 暂时保持均匀，可根据实际分布调整
        self.technique_alpha = torch.ones(self.config.num_techniques).to(self.device)

    def _init_focal_losses(self):
        """初始化Focal Loss实例"""
        # Duration的Focal Loss
        self.duration_focal = FocalLoss(
            alpha=self.duration_alpha,
            gamma=self.gamma,
            reduction='mean',
            ignore_index=-100
        )

        # Fret的Focal Loss（每根弦共享配置）
        self.fret_focal = FocalLoss(
            alpha=self.fret_alpha,
            gamma=self.gamma * 0.8,  # Fret任务gamma稍低
            reduction='mean',
            ignore_index=-100
        )

        # Technique的Focal Loss
        self.technique_focal = FocalLoss(
            alpha=self.technique_alpha,
            gamma=self.gamma * 0.8,  # Technique任务gamma稍低
            reduction='mean',
            ignore_index=-100
        )

        # 也保留标准交叉熵版本用于对比
        self.standard_ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def forward(self, predictions, targets):
        """计算损失"""
        total_loss = 0
        loss_details = {}

        # 1. duration损失
        if 'duration' in predictions and 'duration' in targets:
            pred = predictions['duration']  # [B, L, C]
            target = targets['duration']  # [B, L]

            if self.use_focal:
                duration_loss = self.duration_focal(pred, target)
            else:
                # 使用加权交叉熵
                duration_loss = F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1),
                    weight=self.duration_alpha,
                    ignore_index=-100
                )

            total_loss += duration_loss
            loss_details['duration_loss'] = duration_loss.item()

        # 2. 每根弦的品位损失
        if 'fret' in predictions and 'fret' in targets:
            pred = predictions['fret']  # [B, L, S, C]
            target = targets['fret']  # [B, L, S]
            batch_size, seq_len, num_strings, num_classes = pred.shape

            fret_loss = 0
            for s in range(num_strings):
                pred_s = pred[:, :, s, :]  # [B, L, C]
                target_s = target[:, :, s]  # [B, L]

                if self.use_focal:
                    # 使用Focal Loss
                    string_loss = self.fret_focal(pred_s, target_s)
                else:
                    # 使用加权交叉熵
                    string_loss = F.cross_entropy(
                        pred_s.view(-1, num_classes),
                        target_s.view(-1),
                        weight=self.fret_alpha,
                        ignore_index=-100
                    )

                fret_loss += string_loss

            fret_loss = fret_loss / num_strings
            total_loss += fret_loss
            loss_details['fret_loss'] = fret_loss.item()

        # 3. 每根弦的技巧损失
        if 'technique' in predictions and 'technique' in targets:
            pred = predictions['technique']  # [B, L, S, C]
            target = targets['technique']  # [B, L, S]
            batch_size, seq_len, num_strings, num_classes = pred.shape

            tech_loss = 0
            for s in range(num_strings):
                pred_s = pred[:, :, s, :]  # [B, L, C]
                target_s = target[:, :, s]  # [B, L]

                if self.use_focal:
                    # 使用Focal Loss
                    string_loss = self.technique_focal(pred_s, target_s)
                else:
                    # 使用加权交叉熵
                    string_loss = F.cross_entropy(
                        pred_s.view(-1, num_classes),
                        target_s.view(-1),
                        weight=self.technique_alpha,
                        ignore_index=-100
                    )

                tech_loss += string_loss

            tech_loss = tech_loss / num_strings
            total_loss += tech_loss
            loss_details['technique_loss'] = tech_loss.item()

        loss_details['total_loss'] = total_loss.item()
        return total_loss, loss_details
