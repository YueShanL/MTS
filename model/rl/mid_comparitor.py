import pretty_midi
import numpy as np
from dtw import dtw
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List


class MidiVersionComparator:
    def __init__(self, time_resolution: float = 0.1,
                 feature_type: str = 'chroma',
                 normalize_features: bool = True):
        """
        初始化比较器

        参数:
            time_resolution: 时间分辨率（秒），控制特征提取的精度
            feature_type: 特征类型，可选 'chroma'(色谱), 'pianoroll'(钢琴卷), 'both'(两者结合)
            normalize_features: 是否归一化特征
        """
        self.time_resolution = time_resolution
        self.feature_type = feature_type
        self.normalize_features = normalize_features

        # 验证参数
        valid_features = ['chroma', 'pianoroll', 'both']
        if feature_type not in valid_features:
            raise ValueError(f"feature_type必须是 {valid_features} 之一")

    def extract_features(self, pm: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        从PrettyMIDI对象提取特征矩阵

        返回:
            (time_steps, feature_dim) 形状的特征矩阵
        """
        # 计算总时间步数
        total_time = pm.get_end_time()
        time_steps = int(np.ceil(total_time / self.time_resolution))

        if self.feature_type == 'chroma':
            features = self._extract_chroma_features(pm, time_steps)
        elif self.feature_type == 'pianoroll':
            features = self._extract_pianoroll_features(pm, time_steps)
        else:  # 'both'
            chroma = self._extract_chroma_features(pm, time_steps)
            pianoroll = self._extract_pianoroll_features(pm, time_steps)
            # 合并特征
            features = np.concatenate([chroma, pianoroll], axis=1)

        # 归一化
        if self.normalize_features:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        return features

    def _extract_chroma_features(self, pm: pretty_midi.PrettyMIDI,
                                 time_steps: int) -> np.ndarray:
        """提取色谱特征"""
        # 获取色谱图，转置为 (时间帧, 12)
        chroma = pm.get_chroma(fs=int(1 / self.time_resolution))
        features = chroma.T[:time_steps]

        # 确保维度一致
        if features.shape[0] < time_steps:
            padding = np.zeros((time_steps - features.shape[0], 12))
            features = np.vstack([features, padding])

        return features

    def _extract_pianoroll_features(self, pm: pretty_midi.PrettyMIDI,
                                    time_steps: int) -> np.ndarray:
        """提取钢琴卷特征"""
        feature_dim = 128
        features = np.zeros((time_steps, feature_dim))

        for instrument in pm.instruments:
            for note in instrument.notes:
                start_step = int(note.start / self.time_resolution)
                end_step = int(note.end / self.time_resolution)

                if start_step < time_steps:
                    end_idx = min(end_step, time_steps)
                    # 使用力度作为特征值，归一化到[0,1]
                    features[start_step:end_idx, note.pitch] = np.maximum(
                        features[start_step:end_idx, note.pitch],
                        note.velocity / 127.0
                    )

        return features

    def compare(self, pm1: pretty_midi.PrettyMIDI,
                pm2: pretty_midi.PrettyMIDI,
                use_dtw: bool = True) -> Dict:
        """
        比较两个PrettyMIDI对象

        参数:
            pm1, pm2: PrettyMIDI对象
            use_dtw: 是否使用DTW对齐（True）或直接比较（False）

        返回:
            包含比较结果的字典
        """
        # 提取特征
        X = self.extract_features(pm1)
        Y = self.extract_features(pm2)

        if use_dtw:
            return self._compare_with_dtw(X, Y, pm1, pm2)
        else:
            return self._compare_directly(X, Y, pm1, pm2)

    def _compare_with_dtw(self, X: np.ndarray, Y: np.ndarray,
                          pm1: pretty_midi.PrettyMIDI,
                          pm2: pretty_midi.PrettyMIDI) -> Dict:
        """使用DTW对齐并计算相似度"""

        # 自定义距离函数（余弦距离）
        def cosine_distance(x, y):
            # 处理零向量
            if np.all(x == 0) and np.all(y == 0):
                return 0.0
            if np.all(x == 0) or np.all(y == 0):
                return 1.0
            return cosine(x, y)

        # 执行DTW对齐
        min_distance, cost_matrix, acc_cost_matrix, path = dtw(
            X, Y,
            dist=cosine_distance,
        )
        # path是一个包含两个数组的元组
        path_x, path_y = path

        # 计算路径上的余弦相似度
        similarities = []

        for i, j in zip(path_x, path_y):
            if i < len(X) and j < len(Y):
                sim = 1 - cosine_distance(X[i], Y[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        # 提取基本信息
        basic_info = self._extract_basic_info(pm1, pm2)

        return {
            **basic_info,
            'dtw_distance': min_distance,
            'normalized_distance': min_distance / len(path_x) if len(path_x) > 0 else float('inf'),
            'avg_cosine_similarity': avg_similarity,
            'similarity_std': np.std(similarities) if similarities else 0,
            'path_length': len(path_x),
            'alignment_path': (path_x, path_y),
            'feature_matrices': (X, Y),
            'method': 'dtw_cosine'
        }

    def _compare_directly(self, X: np.ndarray, Y: np.ndarray,
                          pm1: pretty_midi.PrettyMIDI,
                          pm2: pretty_midi.PrettyMIDI) -> Dict:
        """直接比较特征（无时间对齐）"""
        # 将较长的序列截断或填充到与较短的序列相同长度
        min_len = min(len(X), len(Y))
        X_trunc = X[:min_len]
        Y_trunc = Y[:min_len]

        # 计算逐帧余弦相似度
        similarities = []
        for i in range(min_len):
            if not (np.all(X_trunc[i] == 0) and np.all(Y_trunc[i] == 0)):
                sim = 1 - cosine(X_trunc[i], Y_trunc[i])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        basic_info = self._extract_basic_info(pm1, pm2)

        return {
            **basic_info,
            'avg_cosine_similarity': avg_similarity,
            'similarity_std': np.std(similarities) if similarities else 0,
            'frame_count': min_len,
            'method': 'direct_cosine'
        }

    def _extract_basic_info(self, pm1: pretty_midi.PrettyMIDI,
                            pm2: pretty_midi.PrettyMIDI) -> Dict:
        """提取MIDI文件基本信息"""

        def get_notes_info(pm):
            note_count = sum(len(instr.notes) for instr in pm.instruments)
            pitches = [note.pitch for instr in pm.instruments
                       for note in instr.notes]
            durations = [note.end - note.start for instr in pm.instruments
                         for note in instr.notes]

            return {
                'note_count': note_count,
                'pitch_range': (min(pitches) if pitches else 0,
                                max(pitches) if pitches else 0),
                'avg_duration': np.mean(durations) if durations else 0,
                'duration_std': np.std(durations) if durations else 0,
                'instruments': len(pm.instruments)
            }

        info1 = get_notes_info(pm1)
        info2 = get_notes_info(pm2)

        return {
            'duration_ratio': pm2.get_end_time() / pm1.get_end_time()
            if pm1.get_end_time() > 0 else 0,
            'note_count_ratio': info2['note_count'] / info1['note_count']
            if info1['note_count'] > 0 else 0,
            'info1': info1,
            'info2': info2
        }

    def visualize_comparison(self, pm1: pretty_midi.PrettyMIDI,
                             pm2: pretty_midi.PrettyMIDI,
                             save_path: Optional[str] = None) -> Dict:
        """
        Visualize the comparison results

        Returns:
            Comparison result dictionary
        """
        result = self.compare(pm1, pm2, use_dtw=True)
        X, Y = result['feature_matrices']
        path_x, path_y = result['alignment_path']

        fig = plt.figure(figsize=(15, 10))

        # 1. Feature Matrix Visualization
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(X.T, aspect='auto', origin='lower',
                         cmap='Reds', interpolation='nearest')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(Y.T, aspect='auto', origin='lower',
                         cmap='Blues', interpolation='nearest')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 2. DTW Alignment Path
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(path_x, path_y, 'k-', alpha=0.3, linewidth=0.5)

        # Color path points based on similarity
        similarities = []
        for i, j in zip(path_x, path_y):
            if i < len(X) and j < len(Y):
                sim = 1 - cosine(X[i], Y[j]) if not (np.all(X[i] == 0) and np.all(Y[j] == 0)) else 0
                similarities.append(sim)

        if similarities:
            sc = ax3.scatter(path_x[:len(similarities)],
                             path_y[:len(similarities)],
                             c=similarities, s=1, cmap='RdYlGn',
                             vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax3, label='Frame Similarity')

        ax3.grid(True, alpha=0.3)

        # 3. Similarity Distribution
        ax4 = plt.subplot(2, 3, 4)
        if similarities:
            ax4.hist(similarities, bins=30, alpha=0.7, color='green',
                     edgecolor='black')
            ax4.axvline(result['avg_cosine_similarity'], color='red',
                        linestyle='--', linewidth=2,
                        label=f'Average={result["avg_cosine_similarity"]:.3f}')
            ax4.set_title('Similarity Distribution')
            ax4.set_xlabel('Cosine Similarity')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 4. Basic Information Table
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('tight')
        ax5.axis('off')

        info_text = (
            f"Comparison Results:\n"
            f"Method: {result['method']}\n"
            f"Avg Cosine Similarity: {result['avg_cosine_similarity']:.3f}\n"
            f"Similarity Std: {result['similarity_std']:.3f}\n"
            f"DTW Normalized Distance: {result['normalized_distance']:.3f}\n"
            f"Alignment Path Length: {result['path_length']}\n"
            f"Duration Ratio: {result['duration_ratio']:.2f}\n"
            f"Note Count Ratio: {result['note_count_ratio']:.2f}"
        )
        ax5.text(0.1, 0.5, info_text, fontsize=10,
                 verticalalignment='center', fontfamily='monospace')

        # 5. Note Statistics Comparison
        ax6 = plt.subplot(2, 3, 6)
        categories = ['Note Count', 'Pitch Range', 'Average Duration']
        values1 = [
            result['info1']['note_count'],
            result['info1']['pitch_range'][1] - result['info1']['pitch_range'][0],
            result['info1']['avg_duration']
        ]
        values2 = [
            result['info2']['note_count'],
            result['info2']['pitch_range'][1] - result['info2']['pitch_range'][0],
            result['info2']['avg_duration']
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax6.bar(x - width / 2, values1, width, label='Version 1', alpha=0.8)
        ax6.bar(x + width / 2, values2, width, label='Version 2', alpha=0.8)

        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization results saved to: {save_path}")

        ax1.set_title(f'Version1 Features\n{X.shape}')
        ax2.set_title(f'Version2 Features\n{Y.shape}')
        ax3.set_title('DTW Alignment Path')
        ax6.set_xlabel('Metric')
        ax6.set_ylabel('Value')
        ax6.set_title('Basic Statistics Comparison')
        plt.suptitle('MIDI Arrangement Version Comparison')

        plt.show()

        return result


# 使用示例
if __name__ == "__main__":
    # 1. 加载MIDI文件
    pm1 = pretty_midi.PrettyMIDI('out.mid')
    pm2 = pretty_midi.PrettyMIDI('target.mid')

    # 2. 创建比较器（可尝试不同配置）
    comparator = MidiVersionComparator(
        time_resolution=0.1,
        feature_type='chroma',  # 对和声变化敏感
        normalize_features=True
    )

    # 3. 执行比较
    print("正在比较编配版本...")
    result = comparator.compare(pm1, pm2, use_dtw=True)

    # 4. 打印结果摘要
    print(f"\n{'=' * 50}")
    print("比较结果摘要:")
    print(f"{'=' * 50}")
    print(f"平均余弦相似度: {result['avg_cosine_similarity']:.3f}")
    print(f"相似度标准差: {result['similarity_std']:.3f}")
    print(f"DTW归一化距离: {result['normalized_distance']:.3f}")
    print(f"时长比例: {result['duration_ratio']:.2f}")
    print(f"版本1音符数: {result['info1']['note_count']}")
    print(f"版本2音符数: {result['info2']['note_count']}")
    print(f"版本1音高范围: {result['info1']['pitch_range']}")
    print(f"版本2音高范围: {result['info2']['pitch_range']}")

    # 5. 可视化
    print(f"\n生成可视化报告...")
    result = comparator.visualize_comparison(pm1, pm2, save_path='comparison_report.png')

    # 6. 尝试不同特征类型
    print(f"\n{'=' * 50}")
    print("尝试不同特征类型:")

    for feature_type in ['chroma', 'pianoroll', 'both']:
        comparator = MidiVersionComparator(
            time_resolution=0.1,
            feature_type=feature_type
        )
        result = comparator.compare(pm1, pm2, use_dtw=True)
        print(f"{feature_type:10} - 相似度: {result['avg_cosine_similarity']:.3f}")