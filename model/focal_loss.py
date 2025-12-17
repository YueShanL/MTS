import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """增强版焦点损失，支持类别权重、忽略索引和α平衡"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean',
                 weight=None, ignore_index=-100):
        """
        Args:
            alpha: 类别平衡权重，可以是标量或每个类别的权重张量
            gamma: 聚焦参数，gamma越大，对难样本关注越多
            reduction: 'mean'、'sum'或'none'
            weight: 类别权重（类似交叉熵的weight参数）
            ignore_index: 忽略的标签索引
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index

        # 将权重注册为缓冲区，这样它们可以跟随模块移动
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha_tensor', alpha)
        if isinstance(weight, torch.Tensor):
            self.register_buffer('weight_tensor', weight)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits，形状为[N, C]或[B, L, C]
            targets: 目标标签，形状为[N]或[B, L]
        """
        # 确保输入在正确的设备上
        device = inputs.device

        # 获取输入维度
        original_shape = inputs.shape
        if inputs.dim() > 2:
            # 如果输入是多维的（如[B, L, C]），展平为[N, C]
            inputs = inputs.contiguous().view(-1, inputs.size(-1))
            targets = targets.contiguous().view(-1)
        else:
            # 已经是[N, C]
            inputs = inputs.contiguous()
            targets = targets.contiguous()

        # 创建掩码，排除忽略的索引
        mask = (targets != self.ignore_index)
        if not mask.any():
            # 如果没有有效样本，返回0
            if self.reduction == 'mean' or self.reduction == 'sum':
                return torch.tensor(0.0, device=device)
            else:
                return torch.zeros_like(targets, dtype=inputs.dtype, device=device)

        # 只处理有效样本
        valid_inputs = inputs[mask]
        valid_targets = targets[mask]

        # 准备权重参数
        weight = None
        if self.weight is not None:
            if isinstance(self.weight, torch.Tensor):
                # 使用注册的缓冲区，确保在正确的设备上
                if hasattr(self, 'weight_tensor'):
                    weight = self.weight_tensor.to(device)
                else:
                    weight = self.weight.to(device)
            else:
                weight = self.weight

        # 计算交叉熵损失（每个样本）
        ce_loss = F.cross_entropy(
            valid_inputs, valid_targets,
            weight=weight,
            reduction='none'
        )

        # 计算预测概率pt
        probs = F.softmax(valid_inputs, dim=-1)

        # 使用gather获取每个样本的真实类别概率
        pt = torch.exp(-ce_loss)  # 等价于 probs[range(N), valid_targets]

        # 计算调制因子
        focal_weight = (1 - pt) ** self.gamma

        # 应用alpha平衡（如果提供了alpha）
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                # 标量alpha应用到所有类别
                focal_weight = self.alpha * focal_weight
            elif isinstance(self.alpha, torch.Tensor):
                # 每个类别的alpha权重
                if hasattr(self, 'alpha_tensor'):
                    alpha_tensor = self.alpha_tensor.to(device)
                else:
                    alpha_tensor = self.alpha.to(device)
                alpha_t = alpha_tensor[valid_targets]
                focal_weight = alpha_t * focal_weight

        # 计算focal loss
        focal_loss = focal_weight * ce_loss

        # 处理reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            # 需要恢复原始形状
            loss_out = torch.zeros_like(targets, dtype=focal_loss.dtype, device=device)
            loss_out[mask] = focal_loss
            if len(original_shape) > 2:
                # 恢复为原始形状（排除最后一个维度）
                loss_out = loss_out.view(*original_shape[:-1])
            return loss_out
        else:
            raise ValueError(f"不支持的reduction类型: {self.reduction}")