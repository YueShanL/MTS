from typing import Dict, Optional, Tuple

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
from dtw import dtw
from scipy.spatial.distance import cosine
from torch import jit

from model.MTS2.profiler import TimeProfiler


class MidiVersionComparator:
    def __init__(self, time_resolution: float = 0.1,
                 feature_type: str = 'chroma',
                 normalize_features: bool = True,
                 device: Optional[str] = None,
                 enable_profiling: bool = False):  # ← 新增开关
        """
        device: 'cuda' / 'cpu' / None (自动选择)
        enable_profiling: 是否启用性能分析
        """
        self.time_resolution = time_resolution
        self.feature_type = feature_type
        self.normalize_features = normalize_features
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_profiling = enable_profiling

        if self.enable_profiling:
            self.profiler = TimeProfiler()

        valid_features = ['chroma', 'pianoroll', 'both']
        if feature_type not in valid_features:
            raise ValueError(f"feature_type必须是 {valid_features} 之一")

    def extract_features(self, pm: pretty_midi.PrettyMIDI) -> torch.Tensor:
        if self.enable_profiling:
            self.profiler.start('extract_features')

        total_time = pm.get_end_time()
        time_steps = int(np.ceil(total_time / self.time_resolution))

        if self.feature_type == 'chroma':
            feat = self._extract_chroma_features(pm, time_steps)
        elif self.feature_type == 'pianoroll':
            feat = self._extract_pianoroll_features_gpu(pm, time_steps)
        else:  # both
            chroma = self._extract_chroma_features(pm, time_steps)
            pianoroll = self._extract_pianoroll_features_gpu(pm, time_steps)
            feat = torch.cat([chroma, pianoroll], dim=1)

        if self.normalize_features:
            mean = feat.mean(dim=0, keepdim=True)
            std = feat.std(dim=0, keepdim=True)
            feat = (feat - mean) / (std + 1e-8)

        if self.enable_profiling:
            self.profiler.stop('extract_features')
        return feat

    def _extract_chroma_features(self, pm: pretty_midi.PrettyMIDI,
                                 time_steps: int) -> torch.Tensor:
        # CPU 计算，结果转 GPU
        chroma = pm.get_chroma(fs=int(1 / self.time_resolution))
        chroma = chroma.T[:time_steps]
        if chroma.shape[0] < time_steps:
            pad = np.zeros((time_steps - chroma.shape[0], 12))
            chroma = np.vstack([chroma, pad])
        return torch.from_numpy(chroma).float().to(self.device)

    def _extract_pianoroll_features_gpu(self, pm: pretty_midi.PrettyMIDI,
                                        time_steps: int) -> torch.Tensor:
        if self.enable_profiling:
            self.profiler.start('build_pianoroll')

        feat = torch.zeros(time_steps, 128, device=self.device)
        for instr in pm.instruments:
            notes_data = []
            for note in instr.notes:
                start_step = int(note.start / self.time_resolution)
                end_step = int(note.end / self.time_resolution)
                if start_step >= time_steps:
                    continue
                end_step = min(end_step, time_steps)
                notes_data.append((start_step, end_step, note.pitch, note.velocity_logits / 127.0))

            if not notes_data:
                continue

            starts = torch.tensor([d[0] for d in notes_data], device=self.device)
            ends = torch.tensor([d[1] for d in notes_data], device=self.device)
            pitches = torch.tensor([d[2] for d in notes_data], device=self.device)
            velocities = torch.tensor([d[3] for d in notes_data], device=self.device)

            for start, end, pitch, vel in zip(starts, ends, pitches, velocities):
                feat[start:end, pitch] = torch.maximum(feat[start:end, pitch], vel)

        if self.enable_profiling:
            self.profiler.stop('build_pianoroll')
        return feat

    def _cosine_distance_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.enable_profiling:
            self.profiler.start('cosine_dist_mat')

        Xn = F.normalize(X, p=2, dim=1)
        Yn = F.normalize(Y, p=2, dim=1)
        sim = Xn @ Yn.T
        dist = 1 - sim
        dist = torch.clamp(dist, 0, 2)

        # 处理零向量
        zero_mask_X = (X.norm(dim=1) == 0)
        zero_mask_Y = (Y.norm(dim=1) == 0)
        if zero_mask_X.any() or zero_mask_Y.any():
            dist[zero_mask_X, :] = 1.0
            dist[:, zero_mask_Y] = 1.0
            dist[zero_mask_X][:, zero_mask_Y] = 0.0

        if self.enable_profiling:
            self.profiler.stop('cosine_dist_mat')
        return dist

    def _dtw_gpu(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[
        float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.enable_profiling:
            self.profiler.start('dtw_dp')

        D = self._cosine_distance_matrix(X, Y)
        N, M = D.shape
        C = torch.full((N + 1, M + 1), float('inf'), device=self.device)
        C[0, 0] = 0.0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                min_prev = torch.min(C[i - 1, j], C[i, j - 1])
                min_prev = torch.min(min_prev, C[i - 1, j - 1])
                C[i, j] = D[i - 1, j - 1] + min_prev

        # 回溯
        path_x, path_y = [], []
        i, j = N, M
        while i > 0 and j > 0:
            path_x.append(i - 1)
            path_y.append(j - 1)
            if i == 1 and j == 1:
                break
            prev = torch.tensor([C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]], device=self.device)
            min_idx = torch.argmin(prev)
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path_x.reverse()
        path_y.reverse()
        path_x = torch.tensor(path_x, device=self.device)
        path_y = torch.tensor(path_y, device=self.device)

        if self.enable_profiling:
            self.profiler.stop('dtw_dp')
        return C[N, M].item(), C, (path_x, path_y)

    def compare(self, pm1: pretty_midi.PrettyMIDI,
                pm2: pretty_midi.PrettyMIDI,
                use_dtw: bool = True) -> Dict:
        if self.enable_profiling:
            self.profiler.start('total_compare')

        X = self.extract_features(pm1)
        Y = self.extract_features(pm2)

        if use_dtw:
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            return self._compare_with_dtw(X, Y, pm1, pm2)
        else:
            #if len(X) != len(Y):
                #print(f'X({len(X)}) and Y({len(Y)}) are not the same length')
            min_len = min(len(X), len(Y))
            Xt = X[:min_len]
            Yt = Y[:min_len]
            dists = self._cosine_distance_matrix(Xt, Yt).diag()
            sims = 1 - dists
            avg_sim = sims.mean().item()
            norm_dist = None
            path_len = min_len
            method = 'direct_cosine'

        basic_info = self._extract_basic_info(pm1, pm2)

        if self.enable_profiling:
            self.profiler.stop('total_compare')
            self.profiler.report(step='compare')

        return {
            **basic_info,
            'normalized_distance': norm_dist,
            'avg_cosine_similarity': avg_sim,
            'similarity_std': sims.std().item() if len(sims) > 0 else 0,
            'path_length': path_len,
            'method': method,
        }
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
    def evaluate(self, pm1: pretty_midi.PrettyMIDI,
        pm2: pretty_midi.PrettyMIDI,
        use_dtw: bool = True):
        return self.compare(pm1, pm2, use_dtw)['avg_cosine_similarity']

    def _extract_basic_info(self, pm1: pretty_midi.PrettyMIDI,
                            pm2: pretty_midi.PrettyMIDI) -> Dict:
        """提取 MIDI 基本信息（与原类相同）"""

        def get_notes_info(pm):
            note_count = sum(len(instr.notes) for instr in pm.instruments)
            pitches = [note.pitch for instr in pm.instruments for note in instr.notes]
            durations = [note.end - note.start for instr in pm.instruments for note in instr.notes]
            return {
                'note_count': note_count,
                'pitch_range': (min(pitches) if pitches else 0, max(pitches) if pitches else 0),
                'avg_duration': np.mean(durations) if durations else 0,
                'duration_std': np.std(durations) if durations else 0,
                'instruments': len(pm.instruments)
            }

        info1 = get_notes_info(pm1)
        info2 = get_notes_info(pm2)
        return {
            'duration_ratio': pm2.get_end_time() / pm1.get_end_time() if pm1.get_end_time() > 0 else 0,
            'note_count_ratio': info2['note_count'] / info1['note_count'] if info1['note_count'] > 0 else 0,
            'info1': info1,
            'info2': info2
        }


@jit.script
def dtw_core(D: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU 加速的 DTW 核心
    Args:
        D: 距离矩阵 (N, M)
    Returns:
        min_distance, 累积成本矩阵, 路径 x, 路径 y
    """
    N, M = D.shape
    C = torch.full((N + 1, M + 1), float('inf'), dtype=D.dtype, device=D.device)
    C[0, 0] = 0.0

    # 关键：这个双层循环在 TorchScript 中会被编译为高效的 CUDA 代码
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            min_prev = torch.min(C[i - 1, j], C[i, j - 1])
            min_prev = torch.min(min_prev, C[i - 1, j - 1])
            C[i, j] = D[i - 1, j - 1] + min_prev

    # 回溯路径（同样在 GPU 上高效执行）
    max_len = N + M
    path_x = torch.empty(max_len, dtype=torch.long, device=D.device)
    path_y = torch.empty(max_len, dtype=torch.long, device=D.device)
    i, j = N, M
    idx = 0
    while i > 0 and j > 0:
        path_x[idx] = i - 1
        path_y[idx] = j - 1
        idx += 1
        if i == 1 and j == 1:
            break
        # 选择最小前驱
        up = C[i - 1, j]
        left = C[i, j - 1]
        diag = C[i - 1, j - 1]
        if up <= left and up <= diag:
            i -= 1
        elif left <= up and left <= diag:
            j -= 1
        else:
            i -= 1
            j -= 1
    # 反转路径
    path_x = path_x[:idx].flip(0)
    path_y = path_y[:idx].flip(0)
    return C[N, M].item(), C, path_x, path_y

def midi_to_pretty_midi(mido_midi, byte=False):
    """
    将mido MIDI对象转换为pretty_midi对象

    Args:
        mido_midi: mido.MidiFile对象

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    # 方法1: 通过临时文件（简单但效率较低）
    # import tempfile
    # with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
    #     mido_midi.save(tmp.name)
    #     pm = pretty_midi.PrettyMIDI(tmp.name)
    # return pm

    # 方法2: 通过字节流（更高效）
    # 创建一个字节流来保存MIDI数据
    import io
    midi_bytes = io.BytesIO()
    mido_midi.save(file=midi_bytes)
    midi_bytes.seek(0)  # 回到起始位置

    # 使用pretty_midi从字节流加载
    pm = pretty_midi.PrettyMIDI(midi_bytes)
    if byte:
        return pm, midi_bytes
    return pm

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
