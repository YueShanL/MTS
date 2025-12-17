# ============ 配置类 ============
from transformers import PretrainedConfig


class MTSGenConfig(PretrainedConfig):
    """自回归音频到吉他谱生成模型配置（每根弦独立输出）"""

    def __init__(
            self,
            # 音频配置 - 变更为EnCodec常用采样率
            audio_sample_rate=24000,  # 从16000改为24000
            audio_max_length=30,
            audio_channels=1,

            # 编码器配置 - 变更为EnCodec模型
            encoder_model_name="facebook/encodec_24khz",

            # EnCodec输出特征维度（编码器帧特征）
            encoder_output_dim=128,
            # 量化器数量（决定音频质量与特征复杂度）
            num_quantizers=8,

            # Transformer配置
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,

            # 输出配置
            num_strings=6,
            max_fret=24,
            num_durations=13,
            num_techniques=14,

            # 自回归配置
            context_bars=4,
            predict_bars=1,
            notes_per_bar=16,

            # 训练配置
            dropout=0.1,
            layer_norm_eps=1e-5,
            freeze_encoder=False,

            # 时间对齐配置
            audio_feature_rate=50,
            target_temporal_resolution=25,

            **kwargs
    ):
        super().__init__(**kwargs)

        # 自动调整num_attention_heads
        if hidden_size % num_attention_heads != 0:
            for i in range(num_attention_heads, 0, -1):
                if hidden_size % i == 0:
                    num_attention_heads = i
                    break

        # 音频配置
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.audio_channels = audio_channels

        # 编码器配置
        self.encoder_model_name = encoder_model_name

        # EnCodec
        self.encoder_output_dim = encoder_output_dim
        self.num_quantizers = num_quantizers

        # Transformer配置
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        # 输出配置
        self.num_strings = num_strings
        self.max_fret = max_fret
        self.num_durations = num_durations
        self.num_techniques = num_techniques

        # 自回归配置
        self.context_bars = context_bars
        self.predict_bars = predict_bars
        self.notes_per_bar = notes_per_bar

        # 训练配置
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.freeze_encoder = freeze_encoder

        # 时间对齐配置
        self.audio_feature_rate = audio_feature_rate
        self.target_temporal_resolution = target_temporal_resolution

    @staticmethod
    def mtsGen_150m(): return MTSGenConfig(
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
    @staticmethod
    def mtsGen_1b_500m_wide(): return MTSGenConfig(
            hidden_size=2048,  # 从1024翻倍至2048，这是最关键的一步
            num_hidden_layers=24,  # 从12层翻倍至24层
            num_attention_heads=32,  # 头数相应增加，2048 ÷ 32 = 64（每头维度）
            intermediate_size=8192,  # 增加前馈网络维度
            num_durations=13,
            num_techniques=14,
            context_bars=4,
            predict_bars=1,
            max_fret=24,
            freeze_encoder=True
        )

    @staticmethod
    def mtsGen_300m_depth(): return MTSGenConfig(
            num_hidden_layers=48,  # 非常大的深度
            num_attention_heads=32,  # 1600 ÷ 32 = 50
            intermediate_size=6400,  # 增加前馈网络维度
            num_durations=13,
            num_techniques=14,
            context_bars=4,
            predict_bars=1,
            max_fret=24,
            freeze_encoder=True
        )