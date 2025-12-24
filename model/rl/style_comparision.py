import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale


def segment_by_energy(audio_path, method='rms'):
    """
    根据能量（动态）自动分段，识别舒缓/激烈段落
    """
    y, sr = librosa.load(audio_path, sr=22050)

    # 提取能量曲线
    if method == 'rms':
        energy = librosa.feature.rms(y=y)[0]
    elif method == 'loudness':
        S = librosa.stft(y)
        energy = librosa.amplitude_to_db(np.abs(S), ref=np.max).mean(axis=0)

    # 平滑能量曲线（避免微小波动）
    energy_smooth = np.convolve(energy, np.ones(5) / 5, mode='same')

    # 归一化
    energy_norm = minmax_scale(energy_smooth)

    # 动态阈值分割
    high_thresh = 0.7  # 激烈段落阈值
    low_thresh = 0.3  # 舒缓段落阈值

    # 标记段落类型
    segments = []
    in_segment = False
    segment_start = 0
    segment_type = None

    for i in range(len(energy_norm)):
        if energy_norm[i] > high_thresh:
            current_type = 'intense'
        elif energy_norm[i] < low_thresh:
            current_type = 'calm'
        else:
            current_type = 'medium'

        if not in_segment:
            segment_start = i
            segment_type = current_type
            in_segment = True
        elif current_type != segment_type:
            # 段落结束
            segments.append({
                'start_frame': segment_start,
                'end_frame': i - 1,
                'type': segment_type,
                'mean_energy': np.mean(energy_norm[segment_start:i])
            })
            segment_start = i
            segment_type = current_type

    # 添加最后一段
    if in_segment:
        segments.append({
            'start_frame': segment_start,
            'end_frame': len(energy_norm) - 1,
            'type': segment_type,
            'mean_energy': np.mean(energy_norm[segment_start:])
        })

    return segments, energy_norm, y, sr


def extract_segment_features(y, sr, start_frame, end_frame, hop_length=512):
    """
    提取单个段落的表达特征
    """
    # 计算帧到样本的转换
    start_sample = int(start_frame * hop_length)
    end_sample = int(end_frame * hop_length)
    segment_audio = y[start_sample:end_sample]

    features = {}

    if len(segment_audio) < 512:  # 太短的段落跳过
        return None

    # 1. 动态特征（表达强度）
    rms = librosa.feature.rms(y=segment_audio)[0]
    features['dynamics_mean'] = np.mean(rms)
    features['dynamics_std'] = np.std(rms)
    features['dynamics_range'] = np.max(rms) - np.min(rms)

    # 2. 音色表达特征
    mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_variance'] = np.var(mfcc, axis=1)

    # 3. 音高表达特征（情绪）
    try:
        f0, voiced_flag, _ = librosa.pyin(segment_audio,
                                          fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_clean) > 10:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_range'] = np.max(f0_clean) - np.min(f0_clean)
            # 音高变化率（表达激烈程度）
            pitch_slope = np.diff(f0_clean)
            features['pitch_change_rate'] = np.mean(np.abs(pitch_slope))
    except:
        pass

    # 4. 时间表达特征
    # 音符密度（表达紧凑度）
    onset_env = librosa.onset.onset_strength(y=segment_audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
    features['note_density'] = len(onset_frames) / (len(segment_audio) / sr)  # 音符/秒

    # 5. 频谱特征（表达色彩）
    spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
    features['brightness'] = np.mean(spectral_centroid)
    features['brightness_variance'] = np.var(spectral_centroid)

    # 6. 演奏技法特征（针对特定乐器）
    # 例如：吉他揉弦程度、钢琴踏板使用等
    zero_crossing = librosa.feature.zero_crossing_rate(segment_audio)[0]
    features['articulation'] = np.mean(zero_crossing)  # 发音清晰度

    return features


def compare_corresponding_segments(segments1, segments2, y1, y2, sr1, sr2):
    """
    对比两段音频中对应的舒缓/激烈段落
    """
    # 分类段落
    calm_segments_1 = [s for s in segments1 if s['type'] == 'calm']
    intense_segments_1 = [s for s in segments1 if s['type'] == 'intense']

    calm_segments_2 = [s for s in segments2 if s['type'] == 'calm']
    intense_segments_2 = [s for s in segments2 if s['type'] == 'intense']

    # 确保有对应段落
    min_calm = min(len(calm_segments_1), len(calm_segments_2))
    min_intense = min(len(intense_segments_1), len(intense_segments_2))

    # 提取特征并比较
    comparisons = {'calm': [], 'intense': []}

    # 比较舒缓段落
    for i in range(min_calm):
        seg1 = calm_segments_1[i]
        seg2 = calm_segments_2[i]

        feat1 = extract_segment_features(y1, sr1, seg1['start_frame'], seg1['end_frame'])
        feat2 = extract_segment_features(y2, sr2, seg2['start_frame'], seg2['end_frame'])

        if feat1 and feat2:
            similarity = calculate_segment_similarity(feat1, feat2)
            comparisons['calm'].append({
                'segment_pair': (i, i),
                'similarity': similarity,
                'features': (feat1, feat2)
            })

    # 比较激烈段落
    for i in range(min_intense):
        seg1 = intense_segments_1[i]
        seg2 = intense_segments_2[i]

        feat1 = extract_segment_features(y1, sr1, seg1['start_frame'], seg1['end_frame'])
        feat2 = extract_segment_features(y2, sr2, seg2['start_frame'], seg2['end_frame'])

        if feat1 and feat2:
            similarity = calculate_segment_similarity(feat1, feat2)
            comparisons['intense'].append({
                'segment_pair': (i, i),
                'similarity': similarity,
                'features': (feat1, feat2)
            })

    return comparisons


def calculate_segment_similarity(feat1, feat2):
    """
    计算两个段落特征的相似度（加权综合）
    """
    # 定义特征权重（可根据乐器调整）
    weights = {
        'dynamics': 0.25,  # 动态变化
        'pitch': 0.25,  # 音高表达
        'timbre': 0.20,  # 音色特征
        'articulation': 0.15,  # 演奏清晰度
        'brightness': 0.15  # 声音亮度
    }

    similarities = {}

    # 1. 动态相似度
    dyn_keys = ['dynamics_mean', 'dynamics_std', 'dynamics_range']
    dyn_sim = compare_feature_vectors([feat1.get(k, 0) for k in dyn_keys],
                                      [feat2.get(k, 0) for k in dyn_keys])

    # 2. 音高表达相似度
    pitch_keys = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_change_rate']
    pitch_sim = compare_feature_vectors([feat1.get(k, 0) for k in pitch_keys],
                                        [feat2.get(k, 0) for k in pitch_keys])

    # 3. 音色相似度（MFCC）
    if 'mfcc_mean' in feat1 and 'mfcc_mean' in feat2:
        mfcc_sim = cosine_similarity(feat1['mfcc_mean'].reshape(1, -1),
                                     feat2['mfcc_mean'].reshape(1, -1))[0][0]
    else:
        mfcc_sim = 0.5

    # 4. 演奏清晰度
    if 'articulation' in feat1 and 'articulation' in feat2:
        art_sim = 1.0 - abs(feat1['articulation'] - feat2['articulation'])
    else:
        art_sim = 0.5

    # 5. 亮度相似度
    if 'brightness' in feat1 and 'brightness' in feat2:
        bright_sim = 1.0 - abs(feat1['brightness'] - feat2['brightness']) / max(
            feat1['brightness'], feat2['brightness'], 1e-6)
    else:
        bright_sim = 0.5

    # 加权综合
    total_sim = (weights['dynamics'] * dyn_sim +
                 weights['pitch'] * pitch_sim +
                 weights['timbre'] * mfcc_sim +
                 weights['articulation'] * art_sim +
                 weights['brightness'] * bright_sim)

    return {
        'total': total_sim,
        'breakdown': {
            'dynamics': dyn_sim,
            'pitch': pitch_sim,
            'timbre': mfcc_sim,
            'articulation': art_sim,
            'brightness': bright_sim
        }
    }


def compare_feature_vectors(vec1, vec2):
    """计算特征向量相似度（处理缺失值）"""
    valid_idx = [i for i in range(len(vec1)) if vec1[i] != 0 and vec2[i] != 0]
    if not valid_idx:
        return 0.5

    v1 = np.array([vec1[i] for i in valid_idx])
    v2 = np.array([vec2[i] for i in valid_idx])

    # 归一化
    v1_norm = (v1 - np.min(v1)) / (np.max(v1) - np.min(v1) + 1e-6)
    v2_norm = (v2 - np.min(v2)) / (np.max(v2) - np.min(v2) + 1e-6)

    # 余弦相似度
    return np.dot(v1_norm, v2_norm) / (np.linalg.norm(v1_norm) * np.linalg.norm(v2_norm) + 1e-6)


import matplotlib.pyplot as plt


def visualize_expression_comparison(audio1_path, audio2_path):
    """可视化两段音频的表达对比"""
    # 加载并分段
    seg1, energy1, y1, sr1 = segment_by_energy(audio1_path)
    seg2, energy2, y2, sr2 = segment_by_energy(audio2_path)

    # 提取特征并比较
    comparisons = compare_corresponding_segments(seg1, seg2, y1, y2, sr1, sr2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. 能量曲线对比
    time1 = np.linspace(0, len(energy1) / len(y1) * len(y1) / sr1, len(energy1))
    time2 = np.linspace(0, len(energy2) / len(y2) * len(y2) / sr2, len(energy2))

    axes[0].plot(time1, energy1, 'b-', alpha=0.7, label='Audio 1')
    axes[0].plot(time2, energy2, 'r-', alpha=0.7, label='Audio 2')
    axes[0].set_title('Dynamic Energy Comparison')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Normalized Energy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 段落类型对比
    colors = {'calm': 'green', 'medium': 'yellow', 'intense': 'red'}
    for seg in seg1:
        axes[1].axvspan(seg['start_frame'] / len(energy1) * time1[-1],
                        seg['end_frame'] / len(energy1) * time1[-1],
                        alpha=0.3, color=colors[seg['type']])
    axes[1].set_title('Audio 1 Segment Types')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Segment Type')

    for seg in seg2:
        axes[2].axvspan(seg['start_frame'] / len(energy2) * time2[-1],
                        seg['end_frame'] / len(energy2) * time2[-1],
                        alpha=0.3, color=colors[seg['type']])
    axes[2].set_title('Audio 2 Segment Types')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Segment Type')

    plt.tight_layout()

    # 打印相似度结果
    print("=" * 50)
    print("EXPRESSION SIMILARITY REPORT")
    print("=" * 50)

    for segment_type in ['calm', 'intense']:
        if comparisons[segment_type]:
            sim_scores = [c['similarity']['total'] for c in comparisons[segment_type]]
            avg_sim = np.mean(sim_scores)
            print(f"\n{segment_type.upper()} Segments:")
            print(f"  Number of comparable segments: {len(comparisons[segment_type])}")
            print(f"  Average similarity: {avg_sim:.3f}")

            # 打印详细特征相似度
            if len(comparisons[segment_type]) > 0:
                first = comparisons[segment_type][0]['similarity']['breakdown']
                print(f"  Feature breakdown:")
                for feat, score in first.items():
                    print(f"    {feat}: {score:.3f}")

    plt.show()
    return comparisons


# 主程序
if __name__ == "__main__":
    audio1 = "example1.wav"
    audio2 = "example2.wav"

    # 1. 可视化对比
    comparisons = visualize_expression_comparison(audio1, audio2)

    # 2. 详细分析（可选）
    seg1, energy1, y1, sr1 = segment_by_energy(audio1)
    seg2, energy2, y2, sr2 = segment_by_energy(audio2)

    # 提取第一个舒缓段和第一个激烈段进行详细对比
    calm_seg1 = [s for s in seg1 if s['type'] == 'calm'][0]
    intense_seg1 = [s for s in seg1 if s['type'] == 'intense'][0]

    calm_seg2 = [s for s in seg2 if s['type'] == 'calm'][0]
    intense_seg2 = [s for s in seg2 if s['type'] == 'intense'][0]

    # 计算具体特征差异
    print("\n" + "=" * 50)
    print("DETAILED FEATURE COMPARISON")
    print("=" * 50)

    for seg_name, seg1_info, seg2_info in [("Calm", calm_seg1, calm_seg2),
                                           ("Intense", intense_seg1, intense_seg2)]:
        feat1 = extract_segment_features(y1, sr1, seg1_info['start_frame'], seg1_info['end_frame'])
        feat2 = extract_segment_features(y2, sr2, seg2_info['start_frame'], seg2_info['end_frame'])

        if feat1 and feat2:
            print(f"\n{seg_name} Segment:")
            print(f"  Dynamics: Audio1={feat1.get('dynamics_mean', 0):.3f}, "
                  f"Audio2={feat2.get('dynamics_mean', 0):.3f}")
            print(f"  Pitch Range: Audio1={feat1.get('pitch_range', 0):.1f}Hz, "
                  f"Audio2={feat2.get('pitch_range', 0):.1f}Hz")
            print(f"  Note Density: Audio1={feat1.get('note_density', 0):.2f} notes/s, "
                  f"Audio2={feat2.get('note_density', 0):.2f} notes/s")