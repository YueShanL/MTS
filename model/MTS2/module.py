from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, T5Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.t5.modeling_t5 import T5Stack

import basic_pitch_torch.constants
from basic_pitch_torch.model import BasicPitchTorch


# ---------- 配置类 ----------
class MTSGen2Config(PretrainedConfig):
    model_type = "mtsgen2"

    def __init__(
        self,
        freeze_basic_pitch: bool = True,
        sample_rate: int = 22050,
        fft_hop: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_position_embeddings: int = 2000,
        vocab_size: int = 4732,          # 每个头的词汇表大小
        num_heads: int = 6,              # 新增：输出头数量
        decoder_start_token_id: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_basic_pitch = freeze_basic_pitch
        self.sample_rate = sample_rate
        self.fft_hop = fft_hop
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.decoder_start_token_id = decoder_start_token_id


# ---------- 主模型 ----------
class MTSGen2(PreTrainedModel):
    config_class = MTSGen2Config

    def __init__(self, config: MTSGen2Config):
        super().__init__(config)
        self.config = config

        # ---------- Basic Pitch 特征提取 ----------
        self.basic_pitch = BasicPitchTorch()
        if config.freeze_basic_pitch:
            for param in self.basic_pitch.parameters():
                param.requires_grad = False

        # ---------- 特征投影 ----------
        # contour(3) + note + onset
        self.input_proj = nn.Linear(basic_pitch_torch.constants.ANNOTATIONS_N_SEMITONES * 5, config.d_model)

        # ---------- 位置编码 ----------
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

        # ---------- 因果编码器 ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

        # ---------- 解码器（T5Stack，支持 KV 缓存）----------
        t5_config = T5Config(
            d_model=config.d_model,
            d_kv=config.d_model // config.nhead,
            num_layers=config.num_decoder_layers,
            num_heads=config.nhead,
            d_ff=config.dim_feedforward,
            dropout_rate=config.dropout,
            is_decoder=True,
            add_cross_attention=True,
            use_cache=True,
        )
        self.decoder = T5Stack(t5_config)

        # ---------- 共享词嵌入层（每个头使用相同映射）----------
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # ---------- 语言模型头（为每个头独立输出）----------
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size) for _ in range(config.num_heads)
        ])

        # ---------- 初始化权重 ----------
        self.post_init()

    def get_encoder_outputs(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        计算因果编码器输出 (memory)，供生成时使用。
        input_features: [B, T, 264] Basic Pitch 堆叠特征
        returns: [B, T, d_model]
        """
        x = self.input_proj(input_features)                     # [B, T, d_model]
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.position_embeddings(positions)             # [B, T, d_model]
        causal_mask = self._generate_causal_mask(T).to(x.device)
        memory = self.memory_encoder(x, mask=causal_mask)       # [B, T, d_model]
        return memory

    def _prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,          # [B, L, num_heads]
    ) -> torch.Tensor:
        """
        将多个轨道的 token IDs 转换为解码器输入向量。
        对每个时间步，将六个 token 的嵌入向量求和后作为输入。
        """
        # 输入形状检查
        if input_ids.dim() != 3 or input_ids.size(-1) != self.config.num_heads:
            raise ValueError(f"input_ids 形状应为 [batch, seq_len, {self.config.num_heads}]，实际为 {input_ids.shape}")

        # 获取每个 token 的嵌入 [B, L, num_heads, d_model]
        emb = self.embed_tokens(input_ids)  # 通过广播自动处理最后一维
        # 沿轨道维度求和 -> [B, L, d_model]
        combined = emb.sum(dim=2)
        return combined

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        waveform: Optional[torch.Tensor] = None,          # 新增：原始音频波形 [B, 1, T]
        encoder_outputs: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        labels: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # ---------- 处理编码器输出 ----------
        if encoder_outputs is None and waveform is not None:
            # 从波形计算 encoder_outputs
            features = self.extract_features(waveform)            # [B, T, 264]
            encoder_outputs = self.get_encoder_outputs(features)  # [B, T, d_model]
        elif encoder_outputs is None and waveform is None:
            raise ValueError("必须提供 encoder_outputs 或 waveform 之一")

        # ---------- 处理解码器输入 ----------
        assert input_ids is not None, "必须提供 input_ids"
        inputs_embeds = self._prepare_decoder_inputs(input_ids)

        # ---------- 调用解码器 ----------
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = decoder_outputs.last_hidden_state
        new_past = decoder_outputs.past_key_values if use_cache else None

        # ---------- 计算六个头的 logits ----------
        logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            return (loss, logits) + ((new_past,) if use_cache else ())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,          # [B, 1, num_heads] 或 [B, seq_len, num_heads]
        past_key_values: Optional[Tuple] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 如果使用 past_key_values，只取最后一个时间步的 token IDs
        if past_key_values is not None:
            input_ids = input_ids[:, -1:, :]  # 保留最后一个时间步和所有轨道

        return {
            "input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        从原始音频波形提取 Basic Pitch 特征。
        waveforms: [B, T_wave] 或 [B, 1, T_wave] 的音频张量，采样率应与 config.sample_rate 一致。
        returns: 特征张量 [B, T_frame, 264]
        """
        # 确保形状为 [B, T_wave]
        if waveforms.dim() == 3 and waveforms.size(1) == 1:
            waveforms = waveforms.squeeze(1)
        # 调用 BasicPitch 模型（假设输出为 (onset, contour, note) 三个部分）
        # 注：BasicPitchTorch 的具体输出需根据库文档调整
        with torch.no_grad():
            outputs = self.basic_pitch(waveforms)  # 假设返回元组或字典
        # 根据实际输出结构提取三个特征图，并拼接为 [B, T, 264]
        # 示例：若 outputs = (onset, contour, note)，每个形状 [B, T, 88]
        contour = outputs['contour']         # [B, T, 88]
        note    = outputs['note']
        onset   = outputs['onset']
        features = torch.cat([contour, note, onset], dim=-1)   # [B, T, 264]
        return features

    def generate(
        self,
        audio_input: Optional[torch.Tensor] = None,      # 新增：原始音频波形
        max_length: int = 200,
        start_token_ids: Optional[torch.LongTensor] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        **kwargs
    ):
        """
        生成多轨道 token 序列。
        - encoder_outputs: 预计算的编码器输出 [B, T, d_model]（若提供 audio_input 则可省略）
        - audio_input: 原始音频波形 [B, T_wave] 或 [B, 1, T_wave]
        """
        if audio_input is None:
            raise ValueError("必须提供 audio_input")
        if audio_input is not None:
            # 提取特征并计算编码器输出
            features = self.extract_features(audio_input)            # [B, T, 264]
            encoder_outputs = self.get_encoder_outputs(features)    # [B, T, d_model]

        # 后续生成逻辑与之前相同
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        if start_token_ids is None:
            start_token_ids = torch.full(
                (batch_size, self.config.num_heads),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=device
            )
        generated = start_token_ids.unsqueeze(1)  # [B, 1, num_heads]

        past_key_values = None
        for step in range(max_length - 1):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated,
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs,
            )
            outputs = self(**model_inputs)
            logits = outputs.logits          # [B, 1, num_heads, vocab_size]

            next_tokens = []
            for head_idx in range(self.config.num_heads):
                head_logits = logits[:, :, head_idx, :].squeeze(1)  # [B, vocab_size]
                if do_sample:
                    head_logits = head_logits / temperature
                    probs = F.softmax(head_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(head_logits, dim=-1, keepdim=True)
                next_tokens.append(next_token)
            next_tokens = torch.cat(next_tokens, dim=-1).unsqueeze(1)  # [B, 1, num_heads]

            generated = torch.cat([generated, next_tokens], dim=1)
            past_key_values = outputs.past_key_values

        return generated

    def _reorder_cache(self, past_key_values, beam_idx):
        # 支持束搜索（如需）
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    @staticmethod
    def _generate_causal_mask(sz: int) -> torch.Tensor:
        """生成上三角因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# ---------- 使用示例 ----------
if __name__ == "__main__":
    config = MTSGen2Config(
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=6,
        num_heads=6,
    )
    model = MTSGen2(config)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # 模拟一批 wav
    batch_size, seq_len = 2, 20076
    dummy_features = torch.randn(batch_size, seq_len)

    # 2. 生成（默认起始 token ID = 0）
    generated = model.generate(
        dummy_features,
        max_length=79,
    )

    # generated 形状 [2, seq_len, 6]，每个 token ID 范围 0~4367
    print("生成结果形状:", generated.shape)
    print("第一个时间步六个头的 token IDs:", generated[0, 0].tolist())