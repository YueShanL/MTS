import random
from typing import Dict, Optional, Tuple, Union

import guitarpro as gp
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, EncodecModel

from model.dataset import decode
from model.focal_loss import FocalLoss
from model.loss import AutoregressiveMultiTaskLoss
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
        print(f"模型初始化完成")

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
            with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
                # 直接调用编码器获取连续特征，而非使用encode()
                # audio_input形状应为 [B, 1, T]
                audio_features = self.audio_encoder.encoder(audio_input)

                # EnCodec编码器输出形状为 [B, D, T]，需要转置为 [B, T, D]
                if audio_features.dim() == 3:
                    audio_features = audio_features.transpose(1, 2)

            # 确保特征为浮点类型
            audio_features = audio_features.float()

            # 投影到模型隐藏维度
            audio_features = self.audio_projection(audio_features)

            # 时间对齐适配
            return self.temporal_adapter(audio_features)

    def forward(
        self,
        audio_input: torch.Tensor,
        context_notes: Optional[Dict[str, torch.Tensor]] = None,
        target_notes: Optional[Dict[str, torch.Tensor]] = None,
        generate_length: Optional[int] = None,
        teacher_forcing: bool = True,
        do_sample: bool = True,
        return_logits: bool = False,  # 新增参数
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        前向传播，支持返回logits
        
        Args:
            return_logits: 是否返回logits（仅在非Teacher Forcing时有效）
        
        Returns:
            如果teacher_forcing=True或return_logits=False: 返回预测字典
            如果teacher_forcing=False且return_logits=True: 返回元组 (predictions, logits)
        """
        # 编码音频
        audio_features = self.encode_audio(audio_input)
        
        # 编码上下文音符（如果有）
        if context_notes is not None:
            context_embeddings = self.note_embedding(context_notes)
            combined_features = torch.cat([audio_features, context_embeddings], dim=1)
            memory = self.fusion_encoder(combined_features)
        else:
            memory = self.fusion_encoder(audio_features)
        
        # 根据训练模式选择
        if target_notes is not None and teacher_forcing:
            # Teacher Forcing模式：直接返回logits（原本就是logits格式）
            return self._teacher_force_forward(memory, target_notes)
        else:
            # 自回归生成模式
            generate_length = generate_length or self.config.notes_per_bar * self.config.predict_bars
            
            return self._autoregressive_generate(
                memory, generate_length, do_sample=do_sample, return_logits=return_logits
            )


    def _teacher_force_forward(
        self, 
        memory: torch.Tensor, 
        target_notes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Teacher Forcing前向传播，返回logits"""
        batch_size = target_notes['duration'].shape[0]
        target_len = target_notes['duration'].shape[1]
        
        # 准备右移的目标序列
        shifted_notes = {}
        for key in ['duration', 'fret', 'technique']:
            if key in target_notes:
                shifted = torch.full_like(target_notes[key], -100)
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
        
        # 计算输出（logits）
        return self._compute_outputs(decoder_output)  # 返回logits

    def _autoregressive_generate(
        self, 
        memory: torch.Tensor, 
        generate_length: int,
        do_sample: bool = True,
        return_logits: bool = False  # 新增参数，控制是否返回logits
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        自回归生成方法，可选返回logits
        
        Args:
            memory: 编码后的音频特征 [B, T, D]
            generate_length: 生成序列长度
            do_sample: 是否使用采样
            return_logits: 是否返回logits（为True时返回(predictions, logits)元组）
        
        Returns:
            如果return_logits=False: 返回预测字典 {'duration': [B, L], 'fret': [B, L, S], ...}
            如果return_logits=True: 返回元组 (predictions, logits)
        """

        batch_size = memory.shape[0]
        device = memory.device
        
        # 初始化序列
        current_input = self.start_token.expand(batch_size, 1, -1).to(device)
        current_input = current_input + self.positional_encoding.position_embeddings(
            torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        )


        # 存储结果
        all_predictions = {'duration': [], 'fret': [], 'technique': []}
        all_logits = {'duration': [], 'fret': [], 'technique': []} if return_logits else None
        
        # 生成循环
        for step in range(generate_length):
            # 解码当前步
            seq_len = current_input.shape[1]
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)

            
            decoder_output = self.autoregressive_decoder(current_input, memory, tgt_mask)


            last_output = decoder_output[:, -1:, :]
            #last_output = self.output_norm(last_output)
            
            # 计算logits
            step_logits = self._compute_outputs(last_output)
            
            # 存储logits（如果需要）
            if return_logits:
                for key in step_logits:
                    if key in all_logits:
                        all_logits[key].append(step_logits[key])
            
            # 采样预测
            step_predictions = self._sample_next_note(step_logits, do_sample=do_sample)
            
            # 存储预测
            for key in step_predictions:
                if key in all_predictions:
                    all_predictions[key].append(step_predictions[key])
            
            # 准备下一步输入
            next_embedding = self.note_embedding(step_predictions)
            next_embedding = next_embedding + self.positional_encoding.position_embeddings(
                torch.full((batch_size, 1), seq_len, dtype=torch.long, device=device)
            )
            current_input = torch.cat([current_input, next_embedding], dim=1)
        
        # 合并结果
        predictions = {k: torch.cat(v, dim=1) for k, v in all_predictions.items()}
        
        if return_logits:
            logits = {k: torch.cat(v, dim=1) for k, v in all_logits.items()}
            return predictions, logits
        else:
            return predictions

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
                          temperature: float = 0.8,
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


class MixedTrainingForward:
    """混合训练前向传播处理器"""

    def __init__(self, model):
        self.model = model

    def __call__(self, memory, target_notes, teacher_forcing_prob=0.5):
        """执行混合训练前向传播"""
        batch_size = target_notes['duration'].shape[0]
        target_len = target_notes['duration'].shape[1]
        device = memory.device

        # 初始化序列
        current_input = self._init_sequence(batch_size, device)

        # 生成过程
        return self._generate_mixed_sequence(
            memory, target_notes, current_input,
            target_len, teacher_forcing_prob, device
        )

    def _init_sequence(self, batch_size, device):
        """初始化起始序列"""
        start_input = self.model.start_token.expand(batch_size, 1, -1).to(device)
        start_input = start_input + self.model.positional_encoding.position_embeddings(
            torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        )
        return start_input

    def _generate_mixed_sequence(self, memory, target_notes, current_input,
                                 target_len, teacher_forcing_prob, device):
        """混合生成序列"""
        all_outputs = {'duration': [], 'fret': [], 'technique': []}
        all_logits = {'duration': [], 'fret': [], 'technique': []}

        for t in range(target_len):
            # 解码当前步
            step_outputs = self._decode_step(current_input, memory)

            # 存储logits和预测
            self._store_step_outputs(step_outputs, all_outputs, all_logits, t)

            # 准备下一步输入（如果不是最后一步）
            if t < target_len - 1:
                current_input = self._prepare_next_input(
                    current_input, step_outputs, target_notes,
                    t, teacher_forcing_prob, device
                )

        return self._format_outputs(all_outputs, all_logits)

    def _decode_step(self, current_input, memory):
        """解码单个时间步"""
        seq_len = current_input.shape[1]
        tgt_mask = self.model._generate_square_subsequent_mask(seq_len).to(memory.device)

        decoder_output = self.model.autoregressive_decoder(current_input, memory, tgt_mask)
        last_output = decoder_output[:, -1:, :]
        last_output = self.model.output_norm(last_output)

        return self.model._compute_outputs(last_output)

    def _store_step_outputs(self, step_outputs, all_outputs, all_logits, step_idx):
        """存储当前步骤的输出"""
        for key in ['duration', 'fret', 'technique']:
            if key in step_outputs:
                all_logits[key].append(step_outputs[key])
                predicted = self.model._sample_next_note(step_outputs, do_sample=False)[key]
                all_outputs[key].append(predicted)

    def _prepare_next_input(self, current_input, step_outputs, target_notes,
                            step_idx, teacher_forcing_prob, device):
        """准备下一个时间步的输入"""
        use_teacher_forcing = random.random() < teacher_forcing_prob

        if use_teacher_forcing:
            next_note = {k: v[:, step_idx:step_idx + 1] for k, v in target_notes.items()}
        else:
            next_note = self.model._sample_next_note(step_outputs, do_sample=False)

        next_embedding = self.model.note_embedding(next_note)
        seq_len = current_input.shape[1]

        next_embedding = next_embedding + self.model.positional_encoding.position_embeddings(
            torch.full((current_input.shape[0], 1), seq_len, dtype=torch.long, device=device)
        )

        return torch.cat([current_input, next_embedding], dim=1)

    def _format_outputs(self, all_outputs, all_logits):
        """格式化输出"""
        predictions = {k: torch.cat(v, dim=1) for k, v in all_outputs.items() if v}
        logits = {k: torch.cat(v, dim=1) for k, v in all_logits.items() if v}
        return {'predictions': predictions, 'logits': logits}



# ============ 测试代码 ============
def main():
    """测试主函数"""
    config = MTSGenConfig.mtsGen_150m()

    # 创建模型
    model = MTSGen(config)
    model.load_state_dict(torch.load(f'best_model_epoch4.pth'))
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset_path = "../output/Model/dataset"
    dataset = Dataset.load_from_disk(dataset_path).with_format("torch")
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

    dummy_target = sample['target_notes']
    dummy_context = {}
    context = sample['context_notes']
    for key, value in context.items():
        dummy_context[key] = value.unsqueeze(0)

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
        generate_outputs, logits = model(
            audio_input=dummy_audio,
            context_notes=dummy_context,
            teacher_forcing=False,
            generate_length=64,
            do_sample=True,
            return_logits=True
        )

        for key, value in logits.items():
            if value is not None:
                print(f"  {key}: {value.shape}")
    loss = AutoregressiveMultiTaskLoss(config, use_focal=False, device = 'cpu')
    print(loss(logits, dummy_context))
    sample = {}
    for key, value in generate_outputs.items():
        if value is not None:
            sample[key] = value[0]

    song = decode(sample)
    target = decode(dummy_target)

    gp.write(song, 'out.gp5')
    gp.write(target, 'target.gp5')

    print(song)





if __name__ == "__main__":
    main()