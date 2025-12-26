import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class TemporalAdapter(nn.Module):
    def __init__(self, input_dim, target_rate, audio_rate):
        super().__init__()
        self.scale_factor = target_rate / audio_rate

        # 使用插值代替 Upsample
        self.align = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, T, D]
        # 在时间维度上进行插值
        x = x.transpose(1, 2)  # [B, D, T]
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        return x.transpose(1, 2)


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
        '''self.feature_fusion = nn.Sequential(
            nn.Linear(total_embed_dim, total_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(total_embed_dim * 2, config.hidden_size)
        )'''
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_embed_dim, total_embed_dim * 2),
            nn.LayerNorm(total_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_embed_dim * 2, config.hidden_size)
        )

    '''def forward(self, notes: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = notes['duration'].shape

        # 1. 嵌入 duration
        duration_indices = notes['duration'].long()
        pad_mask = (duration_indices == -100)
        duration_indices_clamped = duration_indices.masked_fill(pad_mask, 0)
        duration_emb = self.duration_embedding(duration_indices_clamped)
        duration_emb = duration_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # 2. 嵌入 fret 和 technique
        fret_indices = notes['fret'].long()  # [B, L, 6]
        tech_indices = notes['technique'].long()

        # 创建统一掩码
        string_pad_mask = (fret_indices == -100)  # [B, L, 6]

        # 处理 fret 嵌入
        fret_indices_clamped = fret_indices.masked_fill(string_pad_mask, 0)
        fret_emb_flat = self.fret_embedding(fret_indices_clamped.view(-1))
        fret_emb = fret_emb_flat.view(batch_size, seq_len, self.config.num_strings, -1)
        fret_emb = fret_emb.masked_fill(string_pad_mask.unsqueeze(-1), 0.0)

        # 处理 technique 嵌入
        tech_indices_clamped = tech_indices.masked_fill(string_pad_mask, 0)
        tech_emb_flat = self.technique_embedding(tech_indices_clamped.view(-1))
        tech_emb = tech_emb_flat.view(batch_size, seq_len, self.config.num_strings, -1)
        tech_emb = tech_emb.masked_fill(string_pad_mask.unsqueeze(-1), 0.0)

        # 3. 应用弦间注意力（修正部分）
        # 拼接 fret 和 technique 特征
        string_features = torch.cat([fret_emb, tech_emb], dim=-1)  # [B, L, 6, 64]

        # 重塑为注意力需要的形状: [B*L, 6, 64]
        B, L, S, D = string_features.shape
        string_features = string_features.view(B * L, S, D)

        # 创建注意力掩码
        attn_mask = string_pad_mask.view(B * L, S)  # [B*L, 6]

        # 应用自注意力
        attn_output, _ = self.string_attention(
            string_features, string_features, string_features,
            key_padding_mask=attn_mask
        )

        # 残差连接 + 层归一化
        string_features = string_features + attn_output
        string_features = self.norm1(string_features)

        # 前馈网络
        ffn_output = self.ffn(string_features)
        string_features = string_features + ffn_output
        string_features = self.norm2(string_features)

        # 展平弦维度
        string_features = string_features.view(B, L, -1)  # [B, L, 384]

        # 4. 拼接所有特征
        combined = torch.cat([duration_emb, string_features], dim=-1)  # [B, L, 512]

        # 5. 特征融合
        embeddings = self.feature_fusion(combined)

        return embeddings'''
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

    def __init__(self, hidden_size, max_position=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 正弦余弦位置编码
        pe = torch.zeros(max_position, hidden_size)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_position, hidden_size]
        self.register_buffer('pe', pe)

        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.scale * self.pe[:, :x.size(1)]
        return self.dropout(x)