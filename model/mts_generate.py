import random
from typing import Dict, Optional

import guitarpro as gp
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, EncodecModel

from model.dataset import decode
from model.mts_config import MTSGenConfig
from model.preprocess import TemporalAdapter, NoteEmbedding, PositionalEncoding


# ============ 主模型 ============
class MTSGen(PreTrainedModel):

    config_class = MTSGenConfig

    def __init__(self, config):
        super().__init__(config)

        # 音频编码器
        print(f"加载音频编码器: {config.encoder_model_name}")
        self.audio_encoder = EncodecModel.from_pretrained(config.encoder_model_name)
        # 固定编码器（如果配置要求）
        if config.freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        # =================================================

        # ============ 修改：音频特征投影 ============
        # EnCodec的编码器输出维度是固定的，我们需要从配置或模型中获取
        # 假设我们使用最后一层量化器前的编码器帧特征，其维度为config.encoder_output_dim
        # 如果该配置不存在，则使用一个常见值（例如128）
        #encoder_hidden_size = getattr(config, 'encoder_output_dim', 128)
        #self.audio_projection = nn.Linear(encoder_hidden_size, config.hidden_size)
        # 音频特征投影
        encoder_hidden_size = self.audio_encoder.config.hidden_size
        self.audio_projection = nn.Linear(encoder_hidden_size, config.hidden_size)

        # 时间对齐模块
        self.temporal_adapter = TemporalAdapter(
            input_dim=config.hidden_size,
            target_rate=config.target_temporal_resolution,
            audio_rate=config.audio_feature_rate
        )

        # 音符嵌入层
        self.note_embedding = NoteEmbedding(config)

        # 位置编码
        self.positional_encoding = PositionalEncoding(config.hidden_size)

        # 融合Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers // 2)

        # 自回归Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.autoregressive_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers // 2)

        # 输出归一化
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 输出头
        self.duration_head = nn.Linear(config.hidden_size, config.num_durations)

        # 每根弦独立的输出头
        self.fret_classes_per_string = config.max_fret + 2
        self.fret_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, self.fret_classes_per_string)
            for _ in range(config.num_strings)
        ])

        self.technique_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_techniques)
            for _ in range(config.num_strings)
        ])

        # 起始标记
        self.start_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # 初始化权重
        self.apply(self._init_weights)
        print(f"模型初始化完成，每根弦独立输出品位和技巧")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)

    def encode_audio(self, audio_input: torch.Tensor) -> torch.Tensor:
        """编码音频特征 - 适配EnCodec"""
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            # ============ 方案一：使用编码器的连续特征（推荐） ============
            # 这种方式与Wav2Vec2的输出最相似，都是连续特征
            with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
                # 直接调用编码器获取连续特征，而非使用encode()
                # audio_input形状应为 [B, 1, T]
                audio_features = self.audio_encoder.encoder(audio_input)

                # EnCodec编码器输出形状为 [B, D, T]，需要转置为 [B, T, D]
                if audio_features.dim() == 3:
                    audio_features = audio_features.transpose(1, 2)

            # 确保特征为浮点类型
            audio_features = audio_features.float()
            # ============================================================

            # ============ 方案二：使用量化编码的嵌入（备选） ============
            # 如果你想利用EnCodec的量化特性，可以这样处理：
            # with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            #     # 获取量化编码
            #     encoded_frames = self.audio_encoder.encode(audio_input)
            #
            #     # encoded_frames[0] 是编码列表，每个元素形状为 [B, K, T]
            #     # 其中 K 是量化器数量，T 是时间步
            #     if encoded_frames and len(encoded_frames[0]) > 0:
            #         # 取最后一个量化器的编码（最高质量）
            #         # 或者将所有量化器的编码取平均/拼接
            #         codes = encoded_frames[0][-1]  # [B, 1, T]
            #
            #         # 需要为量化编码创建嵌入层（在__init__中定义）
            #         # self.code_embedding = nn.Embedding(num_codebooks, encoder_output_dim)
            #         audio_features = self.code_embedding(codes)  # [B, T, 1, D]
            #         audio_features = audio_features.squeeze(2)   # [B, T, D]
            # ============================================================

            # 投影到模型隐藏维度
            audio_features = self.audio_projection(audio_features)

            # 时间对齐适配
            return self.temporal_adapter(audio_features)

    def encode_notes(self, context_notes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码上下文音符"""
        note_embeddings = self.note_embedding(context_notes)
        return self.positional_encoding(note_embeddings)

    def forward(
            self,
            audio_input: torch.Tensor,
            context_notes: Optional[Dict[str, torch.Tensor]] = None,
            target_notes: Optional[Dict[str, torch.Tensor]] = None,
            generate_length: Optional[int] = None,
            teacher_forcing: bool = True,
            do_sample=True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码音频
        audio_features = self.encode_audio(audio_input)

        # 编码上下文音符（如果有）
        if context_notes is not None:
            context_embeddings = self.encode_notes(context_notes)
            combined_features = torch.cat([audio_features, context_embeddings], dim=1)
            memory = self.fusion_encoder(combined_features)
        else:
            memory = self.fusion_encoder(audio_features)

        if target_notes is not None and teacher_forcing:
            return self._teacher_force_forward(memory, target_notes)
        else:
            generate_length = generate_length or self.config.notes_per_bar * self.config.predict_bars
            return self._autoregressive_generate(memory, generate_length, do_sample=do_sample)

    def _teacher_force_forward(self, memory: torch.Tensor, target_notes: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        """Teacher forcing前向传播"""
        batch_size = target_notes['duration'].shape[0]
        target_len = target_notes['duration'].shape[1]

        # 准备右移的目标序列，用-100填充
        shifted_notes = {}
        for key in ['duration', 'fret', 'technique']:
            if key in target_notes:
                shifted = torch.full_like(target_notes[key], -100)  # 用-100填充
                shifted[:, 1:] = target_notes[key][:, :-1]
                shifted_notes[key] = shifted

        # 嵌入目标序列
        target_embeddings = self.note_embedding(shifted_notes)
        target_embeddings = self.positional_encoding(target_embeddings)

        # 添加起始标记
        start_tokens = self.start_token.expand(batch_size, -1, -1)
        target_embeddings = torch.cat([start_tokens, target_embeddings], dim=1)

        # 自回归解码
        seq_len = target_embeddings.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(target_embeddings.device)
        decoder_output = self.autoregressive_decoder(target_embeddings, memory, tgt_mask)

        # 去掉起始标记的输出
        decoder_output = decoder_output[:, 1:, :]
        decoder_output = self.output_norm(decoder_output)

        # 计算输出
        return self._compute_outputs(decoder_output)

    def _autoregressive_generate(self, memory: torch.Tensor, generate_length: int,
                                 do_sample=True) -> Dict[str, torch.Tensor]:
        """修复后的自回归生成"""
        batch_size = memory.shape[0]
        device = memory.device

        # 初始化生成序列（起始标记）
        current_input = self.start_token.expand(batch_size, 1, -1).to(device)

        # 存储生成结果
        all_outputs = {'duration': [], 'fret': [], 'technique': []}

        for step in range(generate_length):
            # 为当前输入添加位置编码（只对当前步骤的序列）
            # 注意：这里应该使用绝对位置编码，而不是每次都重新计算
            seq_len = current_input.shape[1]

            # 创建位置ID：0, 1, 2, ..., seq_len-1
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            # 添加位置编码
            current_input_with_pe = current_input + self.positional_encoding.position_embeddings(position_ids)

            # 创建因果掩码
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)

            # 解码
            decoder_output = self.autoregressive_decoder(
                current_input_with_pe, memory, tgt_mask
            )

            # 取最后一个时间步的输出
            last_output = decoder_output[:, -1:, :]
            last_output = self.output_norm(last_output)

            # 计算输出
            step_outputs = self._compute_outputs(last_output)

            # 采样下一个音符
            next_note = self._sample_next_note(step_outputs, do_sample=do_sample)  # 离散值

            # 将离散音符转换为嵌入向量，添加到输入序列
            next_note_embedding = self._discrete_to_embedding(next_note)
            current_input = torch.cat([current_input, next_note_embedding], dim=1)

            # 存储输出
            for key in all_outputs:
                all_outputs[key].append(next_note[key])

        # 合并所有时间步
        outputs = {}
        for key in all_outputs:
            if all_outputs[key]:
                outputs[key] = torch.cat(all_outputs[key], dim=1)

        return outputs

    def _discrete_to_embedding(self, note_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将离散音符转换为嵌入向量"""
        # 创建一个包含所有时间步的字典（当前只有1个时间步）
        batch_size = note_dict['duration'].shape[0]

        # 扩展维度以匹配note_embedding的期望输入
        discrete_notes = {}
        for key in ['duration', 'fret', 'technique']:
            if key in note_dict:
                # note_dict[key] 形状: [B, 1] 或 [B, 1, num_strings]
                discrete_notes[key] = note_dict[key]

        # 通过note_embedding转换为嵌入向量
        embedding = self.note_embedding(discrete_notes)  # [B, 1, hidden_size]

        return embedding

    def _compute_outputs(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算各输出头"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # duration输出
        duration_output = self.duration_head(hidden_states)

        # 每根弦的品位输出
        fret_outputs = []
        for i in range(self.config.num_strings):
            fret_logits = self.fret_heads[i](hidden_states)
            fret_outputs.append(fret_logits)
        fret_output = torch.stack(fret_outputs, dim=2)

        # 每根弦的技巧输出
        technique_outputs = []
        for i in range(self.config.num_strings):
            tech_logits = self.technique_heads[i](hidden_states)
            technique_outputs.append(tech_logits)
        technique_output = torch.stack(technique_outputs, dim=2)

        return {
            'duration': duration_output,
            'fret': fret_output,
            'technique': technique_output
        }

    def _sample_next_note(self, outputs: Dict[str, torch.Tensor],
                          temperature: float = 0.5,
                          top_k: int = 5,
                          top_p: float = 0.9,
                          do_sample: bool = True) -> Dict[str, torch.Tensor]:
        """改进的采样策略（支持温度采样、top-k、top-p）"""
        sampled = {}
        batch_size = outputs['duration'].shape[0]
        device = outputs['duration'].device

        # 采样duration
        if 'duration' in outputs:
            duration_logits = outputs['duration'] / max(temperature, 1e-8)

            if do_sample:
                # 应用top-k过滤
                if top_k > 0:
                    values, indices = torch.topk(duration_logits, min(top_k, duration_logits.shape[-1]), dim=-1)
                    mask = torch.full_like(duration_logits, float('-inf'))
                    mask.scatter_(-1, indices, values)
                    duration_logits = mask

                # 应用top-p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(duration_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    duration_logits = duration_logits.masked_fill(indices_to_remove, float('-inf'))

                # 多项式采样
                probs = F.softmax(duration_logits, dim=-1)
                sampled_duration = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
                sampled_duration = sampled_duration.view(batch_size, -1)
            else:
                # 贪婪采样
                sampled_duration = torch.argmax(duration_logits, dim=-1)

            sampled['duration'] = sampled_duration

        # 采样每根弦的品位（类似处理）
        if 'fret' in outputs:
            fret_logits = outputs['fret'] / max(temperature, 1e-8)

            if do_sample:
                # 对每根弦独立采样
                batch_size, seq_len, num_strings, num_classes = fret_logits.shape
                fret_logits_flat = fret_logits.view(batch_size * seq_len * num_strings, num_classes)

                # 应用top-k和top-p（简化版）
                probs_flat = F.softmax(fret_logits_flat, dim=-1)
                sampled_fret_flat = torch.multinomial(probs_flat, 1)
                sampled_fret = sampled_fret_flat.view(batch_size, seq_len, num_strings)
            else:
                sampled_fret = torch.argmax(fret_logits, dim=-1)

            sampled['fret'] = sampled_fret

        # 采样技巧（类似处理）
        if 'technique' in outputs:
            tech_logits = outputs['technique'] / max(temperature, 1e-8)

            if do_sample:
                batch_size, seq_len, num_strings, num_classes = tech_logits.shape
                tech_logits_flat = tech_logits.view(batch_size * seq_len * num_strings, num_classes)
                probs_flat = F.softmax(tech_logits_flat, dim=-1)
                sampled_tech_flat = torch.multinomial(probs_flat, 1)
                sampled_tech = sampled_tech_flat.view(batch_size, seq_len, num_strings)
            else:
                sampled_tech = torch.argmax(tech_logits, dim=-1)

            sampled['technique'] = sampled_tech

        # 如果duration=0，重置其他特征
        mask = (sampled['duration'] == 0).unsqueeze(-1).expand_as(sampled['fret'])
        if mask.any():
            sampled['fret'][mask] = self.fret_classes_per_string - 1  # 不演奏
            sampled['technique'][mask] = 0  # NORMAL

        return sampled

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


# ============ 损失函数 ============
class AutoregressiveMultiTaskLoss(nn.Module):
    """自回归多任务损失函数（每根弦独立）"""

    def __init__(self, config, weights=None, gamma=2.0):
        super().__init__()
        self.config = config
        self.weights = weights or {'duration': 1.0, 'fret': 1.0, 'technique': 0.5}
        self.gamma = gamma
        self._setup_class_weights()

    def _setup_class_weights(self):
        """设置类别权重"""
        self.class_weights = {
            'duration': torch.ones(self.config.num_durations),
            'fret': torch.ones(self.config.max_fret + 2),
            'technique': torch.ones(self.config.num_techniques)
        }
        self.class_weights['duration'][0] = 0.1  # 无音符权重较低
        self.class_weights['fret'][-1] = 0.1  # 不演奏状态权重较低
        #self.class_weights['technique'][0] = 0.3  # NORMAL技巧权重较低

    def forward(self, predictions, targets, device='cuda'):
        """计算损失"""
        total_loss = 0

        # duration损失
        if 'duration' in predictions and 'duration' in targets:
            pred = predictions['duration']
            target = targets['duration']
            class_weight = self.class_weights['duration'].to(device)
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                target.view(-1),
                weight=class_weight,
                ignore_index=-100  # 忽略填充位置
            )
            total_loss += self.weights['duration'] * loss

        # 每根弦的品位损失
        if 'fret' in predictions and 'fret' in targets:
            pred = predictions['fret']
            target = targets['fret']
            batch_size, seq_len, num_strings, num_classes = pred.shape

            fret_loss = 0
            for s in range(num_strings):
                pred_s = pred[:, :, s, :]
                target_s = target[:, :, s]
                class_weight = self.class_weights['fret'].to(device)
                loss = F.cross_entropy(
                    pred_s.view(-1, num_classes),
                    target_s.view(-1),
                    weight=class_weight,
                    ignore_index=-100
                )
                pt = torch.exp(-loss)  # 模型对真实类别的预测概率
                focal_loss = ((1 - pt) ** self.gamma) * loss
                fret_loss += focal_loss

            total_loss += self.weights['fret'] * (fret_loss / num_strings)

        # 每根弦的技巧损失
        if 'technique' in predictions and 'technique' in targets:
            pred = predictions['technique']
            target = targets['technique']
            batch_size, seq_len, num_strings, num_classes = pred.shape

            tech_loss = 0
            for s in range(num_strings):
                pred_s = pred[:, :, s, :]
                target_s = target[:, :, s]
                class_weight = self.class_weights['technique'].to(device)
                loss = F.cross_entropy(
                    pred_s.view(-1, num_classes),
                    target_s.view(-1),
                    weight=class_weight,
                    ignore_index=-100
                )
                tech_loss += loss

            total_loss += self.weights['technique'] * (tech_loss / num_strings)

        return total_loss


class MTSGenTrainer:

    def __init__(self, config=None, model = None):
        self.config = config or MTSGenConfig()
        self.model = model if model is not None else MTSGen(self.config)
        self.loss_fn = AutoregressiveMultiTaskLoss(self.config)
        self._setup_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
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
            audio_list.append(sample['audio_input'])

            # 2. 上下文数据 - 直接提取，假设结构固定
            context_notes = sample['context_notes']
            context_duration.append(context_notes['duration'])
            context_fret.append(context_notes['fret'])
            context_technique.append(context_notes['technique'])

            # 3. 目标数据
            target_notes = sample['target_notes']
            target_duration.append(target_notes['duration'])
            target_fret.append(target_notes['fret'])
            target_technique.append(target_notes['technique'])

        # 使用torch.stack一次性堆叠，减少碎片化操作
        batched = {
            'audio_input': torch.stack(audio_list).unsqueeze(1),  # [B, 1, T]
            'context_notes': {
                'duration': torch.stack(context_duration),
                'fret': torch.stack(context_fret),
                'technique': torch.stack(context_technique)
            },
            'target_notes': {
                'duration': torch.stack(target_duration),
                'fret': torch.stack(target_fret),
                'technique': torch.stack(target_technique)
            }
        }

        return batched

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



# ============ 测试代码 ============
def main():
    """测试主函数"""
    config = MTSGenConfig(
        hidden_size=1024,  # 增加隐藏层维度
        num_hidden_layers=12,  # 增加层数
        num_attention_heads=16,  # 增加头数，1024 ÷ 16 = 64
        intermediate_size=4096,  # 增加前馈网络维度
        num_durations=13,
        num_techniques=14,
        context_bars=4,
        predict_bars=1,
        max_fret=24,
        freeze_encoder=True
    )

    # 创建模型
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'checkpoint_epoch_20.pt')['model_state_dict'])
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset_path = "../output/Model/dataset"
    dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
    #dataset = AudioGuitarTabDataset(dataset['audio_input'], dataset['target_notes'])
    sample = dataset[random.randint(0, len(dataset) - 1)]

    # 测试数据
    batch_size = 1
    audio_length = 24000 * 8
    context_length = 64

    # 模拟音频输入
    dummy_audio = torch.randn(batch_size,1, audio_length)
    dummy_audio = sample['audio_input'][:audio_length].unsqueeze(0).unsqueeze(1)
    import soundfile as sf

    sf.write(f"out.wav",  dummy_audio.squeeze().cpu().numpy(), 24000)


    # 模拟上下文音符
    dummy_context = {
        'duration': torch.randint(0, 13, (batch_size, context_length)),
        'fret': torch.randint(0, 26, (batch_size, context_length, 6)),
        'technique': torch.randint(0, 14, (batch_size, context_length, 6))
    }


    # 模拟目标音符
    target_length = 16
    dummy_target = {
        'duration': torch.randint(0, 13, (batch_size, target_length)),
        'fret': torch.randint(0, 26, (batch_size, target_length, 6)),
        'technique': torch.randint(0, 14, (batch_size, target_length, 6))
    }
    dummy_target = {}
    target = sample['target_notes']
    for key, value in target.items():
        dummy_target[key] = value[:64]

    # 测试teacher forcing模式
    '''print("\n测试Teacher Forcing模式:")
    outputs = model(
        audio_input=dummy_audio,
        context_notes=dummy_context,
        target_notes=dummy_target,
        teacher_forcing=True
    )

    for key, value in outputs.items():
        if value is not None:
            print(f"  {key}: {value.shape}")'''

    # 测试生成模式
    print("\n测试生成模式:")
    model.eval()
    with torch.no_grad():
        generate_outputs = model(
            audio_input=dummy_audio,
            #context_notes=dummy_context,
            teacher_forcing=False,
            generate_length=64,
            do_sample=True
        )

        for key, value in generate_outputs.items():
            if value is not None:
                print(f"  {key}: {value.shape}")

    sample = {}
    for key, value in generate_outputs.items():
        if value is not None:
            sample[key] = value[0]

    song = decode(sample)
    target = decode(dummy_target)

    gp.write(song, 'out.gp5')
    gp.write(target, 'target.gp5')

    print(song)


    '''# ============ 新增：损失计算案例 ============
    print("\n" + "=" * 50)
    print("损失计算案例")
    print("=" * 50)

    # 创建损失函数实例
    loss_fn = AutoregressiveMultiTaskLoss(config)
    print(f"损失函数权重: {loss_fn.weights}")

    # 计算训练损失
    print("\n1. 计算Teacher Forcing模式的损失:")
    train_loss = loss_fn(outputs, dummy_target, device='cpu')
    print(f"  总损失: {train_loss.item():.4f}")

    # 分析各部分的损失
    print("\n2. 分析各任务损失:")

    # duration损失
    duration_pred = outputs['duration']  # [2, 16, 13]
    duration_target = dummy_target['duration']  # [2, 16]
    duration_loss = F.cross_entropy(
        duration_pred.view(-1, duration_pred.size(-1)),
        duration_target.view(-1),
        weight=loss_fn.class_weights['duration']
    )
    print(f"  duration损失: {duration_loss.item():.4f}")

    # 每根弦的fret损失
    fret_pred = outputs['fret']  # [2, 16, 6, 26]
    fret_target = dummy_target['fret']  # [2, 16, 6]
    batch_size, seq_len, num_strings, num_classes = fret_pred.shape
    fret_loss_sum = 0
    for s in range(num_strings):
        pred_s = fret_pred[:, :, s, :]  # [2, 16, 26]
        target_s = fret_target[:, :, s]  # [2, 16]
        loss = F.cross_entropy(
            pred_s.view(-1, num_classes),
            target_s.view(-1),
            weight=loss_fn.class_weights['fret']
        )
        fret_loss_sum += loss.item()
    fret_loss_avg = fret_loss_sum / num_strings
    print(f"  fret损失(平均每弦): {fret_loss_avg:.4f}")

    # 每根弦的technique损失
    technique_pred = outputs['technique']  # [2, 16, 6, 14]
    technique_target = dummy_target['technique']  # [2, 16, 6]
    batch_size, seq_len, num_strings, num_classes = technique_pred.shape
    technique_loss_sum = 0
    for s in range(num_strings):
        pred_s = technique_pred[:, :, s, :]  # [2, 16, 14]
        target_s = technique_target[:, :, s]  # [2, 16]
        loss = F.cross_entropy(
            pred_s.view(-1, num_classes),
            target_s.view(-1),
            weight=loss_fn.class_weights['technique']
        )
        technique_loss_sum += loss.item()
    technique_loss_avg = technique_loss_sum / num_strings
    print(f"  technique损失(平均每弦): {technique_loss_avg:.4f}")

    # 验证损失函数计算的正确性
    print("\n3. 验证损失函数计算:")
    print(
        f"  手动计算加权总损失: {duration_loss.item() * loss_fn.weights['duration'] + fret_loss_avg * loss_fn.weights['fret'] + technique_loss_avg * loss_fn.weights['technique']:.4f}")
    print(f"  损失函数输出: {train_loss.item():.4f}")

    # 测试不同权重配置
    print("\n4. 测试不同权重配置:")
    custom_weights = {
        'duration': 2.0,  # 提高duration权重
        'fret': 1.0,
        'technique': 0.3  # 降低technique权重
    }
    custom_loss_fn = AutoregressiveMultiTaskLoss(config, weights=custom_weights)
    custom_loss = custom_loss_fn(outputs, dummy_target, device='cpu')
    print(f"  自定义权重损失: {custom_loss.item():.4f}")

    # 测试处理无效目标值
    print("\n5. 测试处理无效目标值:")
    # 创建一个包含无效值的目标张量
    invalid_target = dummy_target.copy()
    invalid_target['duration'][0, 0] = -100  # 使用ignore_index
    invalid_target['fret'][0, 0, 0] = -100
    invalid_target['technique'][0, 0, 0] = -100

    # 使用忽略索引的损失函数
    ce_loss_with_ignore = nn.CrossEntropyLoss(ignore_index=-100)

    # duration损失（忽略无效值）
    duration_loss_ignore = ce_loss_with_ignore(
        duration_pred.view(-1, duration_pred.size(-1)),
        invalid_target['duration'].view(-1)
    )
    print(f"  忽略无效值的duration损失: {duration_loss_ignore.item():.4f}")

    # 显示一些预测和目标的对比
    print("\n6. 预测与目标对比（第一个样本的前5个时间步）:")
    with torch.no_grad():
        # 采样预测结果
        sampled_pred = model._sample_next_note(outputs)

        print(f"  Duration预测: {sampled_pred['duration'][0, :5].tolist()}")
        print(f"  Duration目标: {dummy_target['duration'][0, :5].tolist()}")

        print(f"  Fret预测（第一弦）: {sampled_pred['fret'][0, :5, 0].tolist()}")
        print(f"  Fret目标（第一弦）: {dummy_target['fret'][0, :5, 0].tolist()}")

        print(f"  Technique预测（第一弦）: {sampled_pred['technique'][0, :5, 0].tolist()}")
        print(f"  Technique目标（第一弦）: {dummy_target['technique'][0, :5, 0].tolist()}")

    # 测试训练管道
    print("\n7. 测试训练管道:")
    pipeline = AutoregressiveTrainingPipeline(config)

    # 创建一个模拟的训练批次
    train_batch = {
        'audio_input': dummy_audio,
        'context_notes': dummy_context,
        'target_notes': dummy_target
    }

    # 执行一个训练步骤
    step_loss = pipeline.train_step(train_batch)
    print(f"  训练步骤损失: {step_loss:.4f}")

    # 检查模型参数是否更新
    print("  检查参数梯度:")
    for name, param in pipeline.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"    {name}: 梯度范数 = {grad_norm:.6f}")

    print("\n" + "=" * 50)
    print("损失计算案例完成")
    print("=" * 50)'''



if __name__ == "__main__":
    main()