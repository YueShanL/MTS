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

        # 计算总嵌入维度
        total_embed_dim = (
                self.duration_embed_dim +  # duration: 128
                self.per_string_embed_dim * config.num_strings +  # 所有弦的品位: 32*6=192
                self.per_string_embed_dim * config.num_strings  # 所有弦的技巧: 32*6=192
        )

        print(f"NoteEmbeddingSimple维度: total={total_embed_dim}, hidden_size={config.hidden_size}")

        # 线性层融合所有特征
        self.feature_fusion = nn.Linear(total_embed_dim, config.hidden_size)

    def forward(self, notes: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = notes['duration'].shape

        # 嵌入duration - 处理填充值（-100）
        duration_indices = notes['duration'].long()
        # 创建一个掩码，标记填充位置
        pad_mask = (duration_indices == -100)
        # 将填充位置替换为0（临时，用于嵌入查找）
        duration_indices_clamped = duration_indices.clone()
        duration_indices_clamped[pad_mask] = 0
        duration_emb = self.duration_embedding(duration_indices_clamped)
        # 将填充位置的嵌入置为零
        duration_emb[pad_mask] = 0.0

        # 嵌入所有弦的品位并拼接
        fret_emb_list = []
        for s in range(self.config.num_strings):
            fret_idx = notes['fret'][:, :, s].long()
            pad_mask_fret = (fret_idx == -100)
            fret_idx_clamped = fret_idx.clone()
            fret_idx_clamped[pad_mask_fret] = 0
            emb = self.fret_embedding(fret_idx_clamped)
            emb[pad_mask_fret] = 0.0
            fret_emb_list.append(emb)
        fret_emb = torch.cat(fret_emb_list, dim=-1)

        # 嵌入所有弦的技巧并拼接
        technique_emb_list = []
        for s in range(self.config.num_strings):
            tech_idx = notes['technique'][:, :, s].long()
            pad_mask_tech = (tech_idx == -100)
            tech_idx_clamped = tech_idx.clone()
            tech_idx_clamped[pad_mask_tech] = 0
            emb = self.technique_embedding(tech_idx_clamped)
            emb[pad_mask_tech] = 0.0
            technique_emb_list.append(emb)
        technique_emb = torch.cat(technique_emb_list, dim=-1)

        # 拼接所有特征
        combined = torch.cat([duration_emb, fret_emb, technique_emb], dim=-1)

        # 融合特征
        embeddings = self.feature_fusion(combined)

        return embeddings


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)