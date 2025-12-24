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
        """设置吉他任务特定的权重"""
        # Duration
        self.duration_alpha = torch.ones(self.config.num_durations).to(self.device)
        self.duration_alpha[0] = 1.5  # 休止符权重最高

        # Fret
        self.fret_alpha = torch.ones(self.config.max_fret + 2).to(self.device)
        self.fret_alpha[-1] = 0.3


        # Technique
        self.technique_alpha = torch.ones(self.config.num_techniques).to(self.device)
        #self.technique_alpha[0] = 0.6

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
