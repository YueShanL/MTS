from typing import Dict

import torch
from torch import nn


class TemporalAdapter(nn.Module):
    """时间对齐适配器"""

    def __init__(self, input_dim, target_rate, audio_rate):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=target_rate / audio_rate, mode='linear')

    def forward(self, x):
        return self.upsample(x.transpose(1, 2)).transpose(1, 2)


class NoteEmbedding(nn.Module):
    """音符特征嵌入层（每根弦独立，保留弦间差异）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 使用固定的维度分配
        self.duration_embed_dim = 128
        self.per_string_embed_dim = 32  # 每根弦的嵌入维度

        # 时值嵌入
        self.duration_embedding = nn.Embedding(config.num_durations, self.duration_embed_dim)

        # 每根弦的品位嵌入
        self.fret_classes_per_string = config.max_fret + 2
        self.fret_embedding = nn.Embedding(self.fret_classes_per_string, self.per_string_embed_dim)

        # 每根弦的技巧嵌入
        self.technique_embedding = nn.Embedding(config.num_techniques, self.per_string_embed_dim)

        # 弦间注意力机制
        self.string_attention = nn.MultiheadAttention(
            embed_dim=self.per_string_embed_dim * 2,  # 品位+技巧的拼接维度
            num_heads=4,  # 4个注意力头
            batch_first=True,  # 批次维度在前
            dropout=0.1  # 防止过拟合
        )

        # 注意力后的层归一化和前馈网络
        self.norm1 = nn.LayerNorm(self.per_string_embed_dim * 2)
        self.norm2 = nn.LayerNorm(self.per_string_embed_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(self.per_string_embed_dim * 2, self.per_string_embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.per_string_embed_dim * 4, self.per_string_embed_dim * 2)
        )

        # 计算总嵌入维度
        total_embed_dim = (
                self.duration_embed_dim +  # duration: 128
                self.per_string_embed_dim * config.num_strings * 2  # 注意力后的弦特征: 32*6*2=384
        )

        print(f"NoteEmbeddingSimple维度: total={total_embed_dim}, hidden_size={config.hidden_size}")

        # 线性层融合所有特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_embed_dim, total_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(total_embed_dim * 2, config.hidden_size)
        )

    def forward(self, notes: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = notes['duration'].shape

        # 嵌入duration
        duration_indices = notes['duration'].long()
        pad_mask = (duration_indices == -100)
        duration_indices_clamped = duration_indices.clone()
        duration_indices_clamped[pad_mask] = 0
        duration_emb = self.duration_embedding(duration_indices_clamped)
        duration_emb[pad_mask] = 0.0

        # 向量化处理所有弦的品位
        # fret形状: [B, L, num_strings]
        fret_indices = notes['fret'].long()  # [B, L, 6]
        fret_pad_mask = (fret_indices == -100)
        fret_indices_clamped = fret_indices.clone()
        fret_indices_clamped[fret_pad_mask] = 0

        # 重塑为 [B*L*6] 以进行批量嵌入查找
        fret_indices_flat = fret_indices_clamped.view(-1)
        fret_emb_flat = self.fret_embedding(fret_indices_flat)  # [B*L*6, 32]
        fret_emb = fret_emb_flat.view(batch_size, seq_len, self.config.num_strings, -1)  # [B, L, 6, 32]

        # 将填充位置的嵌入置零
        fret_emb[fret_pad_mask.unsqueeze(-1).expand_as(fret_emb)] = 0.0

        # 展平弦维度 [B, L, 6*32]
        fret_emb = fret_emb.view(batch_size, seq_len, -1)

        # 向量化处理所有弦的技巧
        tech_indices = notes['technique'].long()
        tech_pad_mask = (tech_indices == -100)
        tech_indices_clamped = tech_indices.clone()
        tech_indices_clamped[tech_pad_mask] = 0

        tech_indices_flat = tech_indices_clamped.view(-1)
        tech_emb_flat = self.technique_embedding(tech_indices_flat)
        tech_emb = tech_emb_flat.view(batch_size, seq_len, self.config.num_strings, -1)
        tech_emb[tech_pad_mask.unsqueeze(-1).expand_as(tech_emb)] = 0.0
        tech_emb = tech_emb.view(batch_size, seq_len, -1)

        # 拼接所有特征
        combined = torch.cat([duration_emb, fret_emb, tech_emb], dim=-1)

        # 融合特征
        embeddings = self.feature_fusion(combined)

        return embeddings


class PositionalEncoding(nn.Module):
    """绝对位置编码"""

    def __init__(self, hidden_size, max_position=512):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.max_position = max_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """为输入添加位置编码"""
        batch_size, seq_len, hidden_size = x.shape

        # 创建位置ID
        position_ids = torch.arange(seq_len, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 获取位置编码
        position_embeddings = self.position_embeddings(position_ids)

        # 添加到输入
        return x + position_embeddings