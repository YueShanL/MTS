#!/usr/bin/env python3
"""
MIDI相似度比较测试脚本
测试各种比较方法的效果和性能
"""

import os
import time

from model.mid_comparsion import create_sample_midi, MIDISimilarityToolkit


def test_basic_comparison(mid1, mid2):
    """测试基本比较功能"""
    print("=" * 60)
    print("MIDI相似度比较工具包 - 基本测试")
    print("=" * 60)

    # 测试各种比较方法
    methods = ['exact', 'balanced', 'style', 'fast', 'melodic']

    results = {}
    for method in methods:
        print(f"\n使用 '{method}' 方法比较:")
        comparator = MIDISimilarityToolkit(method=method)

        # 测试性能
        start_time = time.time()
        similarity = comparator.compare(mid1, mid2)
        elapsed = time.time() - start_time

        results[method] = {
            'similarity': similarity,
            'time': elapsed
        }

        print(f"  相似度: {similarity:.4f}")
        print(f"  耗时: {elapsed:.4f}秒")

    # 显示总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)

    for method, result in results.items():
        print(f"{method:10s} - 相似度: {result['similarity']:.4f}, 耗时: {result['time']:.4f}秒")

    return results


def test_performance_scaling():
    """测试性能扩展性"""
    print("\n" + "=" * 60)
    print("性能扩展性测试")
    print("=" * 60)

    # 创建多个测试文件
    test_files = []
    for i in range(5):
        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # 添加随机音符
        for j in range(50):
            note = pretty_midi.Note(
                velocity=np.random.randint(60, 100),
                pitch=np.random.randint(48, 72),
                start=j * 0.2,
                end=j * 0.2 + 0.15
            )
            piano.notes.append(note)

        midi.instruments.append(piano)

        # 保存到临时文件
        import tempfile
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f'test_perf_{i}.mid')
        midi.write(file_path)
        test_files.append(file_path)

    # 测试批量比较性能
    comparator = MIDISimilarityToolkit(method='fast')

    print("测试批量比较性能...")
    start_time = time.time()

    # 创建配对：每个文件与第一个文件比较
    pairs_a = [test_files[0]] * 4
    pairs_b = test_files[1:]

    similarities = comparator.batch_compare(pairs_a, pairs_b)
    elapsed = time.time() - start_time

    print(f"批量比较 {len(pairs_a)} 对文件耗时: {elapsed:.4f}秒")
    print(f"平均每对耗时: {elapsed / len(pairs_a):.4f}秒")

    for i, sim in enumerate(similarities):
        print(f"  文件1 vs 文件{i + 2}: {sim:.4f}")

    # 清理临时文件
    for file in test_files:
        try:
            os.remove(file)
        except:
            pass


def test_method_comparison():
    """比较不同方法的差异"""
    print("\n" + "=" * 60)
    print("不同比较方法对比")
    print("=" * 60)

    # 创建三组不同相似度的MIDI文件
    # 组1: 高度相似（相同旋律，不同音色）
    # 组2: 中等相似（相似旋律，不同节奏）
    # 组3: 低相似度（不同旋律）

    import tempfile
    temp_dir = tempfile.gettempdir()

    test_cases = []

    # 用例1: 高度相似
    print("创建测试用例1: 高度相似...")
    midi1 = pretty_midi.PrettyMIDI()
    midi2 = pretty_midi.PrettyMIDI()

    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano1 = pretty_midi.Instrument(program=piano_program)
    piano2 = pretty_midi.Instrument(program=piano_program)

    # 相同旋律
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, pitch in enumerate(pitches):
        note1 = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=i * 0.5 + 0.4
        )
        note2 = pretty_midi.Note(
            velocity=95,  # 力度稍有不同
            pitch=pitch,
            start=i * 0.5 + 0.01,  # 时间稍有不同
            end=i * 0.5 + 0.39
        )
        piano1.notes.append(note1)
        piano2.notes.append(note2)

    midi1.instruments.append(piano1)
    midi2.instruments.append(piano2)

    path1 = os.path.join(temp_dir, 'high_sim1.mid')
    path2 = os.path.join(temp_dir, 'high_sim2.mid')
    midi1.write(path1)
    midi2.write(path2)
    test_cases.append(('高度相似', path1, path2))

    # 用例2: 中等相似
    print("创建测试用例2: 中等相似...")
    midi3 = pretty_midi.PrettyMIDI()
    midi4 = pretty_midi.PrettyMIDI()

    piano3 = pretty_midi.Instrument(program=piano_program)
    piano4 = pretty_midi.Instrument(program=piano_program)

    # 相似旋律，不同节奏
    for i, pitch in enumerate(pitches):
        note3 = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=i * 0.5 + 0.4
        )
        note4 = pretty_midi.Note(
            velocity=100,
            pitch=pitch - 2,  # 移调
            start=i * 0.6,  # 不同节奏
            end=i * 0.6 + 0.3
        )
        piano3.notes.append(note3)
        piano4.notes.append(note4)

    midi3.instruments.append(piano3)
    midi4.instruments.append(piano4)

    path3 = os.path.join(temp_dir, 'mid_sim1.mid')
    path4 = os.path.join(temp_dir, 'mid_sim2.mid')
    midi3.write(path3)
    midi4.write(path4)
    test_cases.append(('中等相似', path3, path4))

    # 用例3: 低相似度
    print("创建测试用例3: 低相似度...")
    midi5 = pretty_midi.PrettyMIDI()
    midi6 = pretty_midi.PrettyMIDI()

    piano5 = pretty_midi.Instrument(program=piano_program)
    piano6 = pretty_midi.Instrument(program=piano_program)

    # 不同旋律
    pitches1 = [60, 62, 64, 65, 67, 69, 71, 72]
    pitches2 = [48, 50, 52, 53, 55, 57, 59, 60]

    for i in range(8):
        note5 = pretty_midi.Note(
            velocity=100,
            pitch=pitches1[i],
            start=i * 0.5,
            end=i * 0.5 + 0.4
        )
        note6 = pretty_midi.Note(
            velocity=100,
            pitch=pitches2[i],
            start=i * 0.7,
            end=i * 0.7 + 0.5
        )
        piano5.notes.append(note5)
        piano6.notes.append(note6)

    midi5.instruments.append(piano5)
    midi6.instruments.append(piano6)

    path5 = os.path.join(temp_dir, 'low_sim1.mid')
    path6 = os.path.join(temp_dir, 'low_sim2.mid')
    midi5.write(path5)
    midi6.write(path6)
    test_cases.append(('低相似度', path5, path6))

    # 测试所有方法
    methods = ['exact', 'balanced', 'style', 'fast', 'melodic']
    results = {method: [] for method in methods}

    for case_name, path1, path2 in test_cases:
        print(f"\n测试用例: {case_name}")
        for method in methods:
            comparator = MIDISimilarityToolkit(method=method)
            similarity = comparator.compare(path1, path2)
            results[method].append(similarity)
            print(f"  {method:10s}: {similarity:.4f}")

    # 可视化结果
    visualize_results(results, test_cases)

    # 清理临时文件
    for _, path1, path2 in test_cases:
        try:
            os.remove(path1)
            os.remove(path2)
        except:
            pass


def visualize_results(results, test_cases):
    """可视化比较结果"""
    case_names = [case[0] for case in test_cases]
    methods = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(case_names))
    width = 0.15
    multiplier = 0

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, method in enumerate(methods):
        offset = width * multiplier
        ax.bar(x + offset, results[method], width, label=method, color=colors[i])
        multiplier += 1

    ax.set_ylabel('相似度分数')
    ax.set_title('不同比较方法在各测试用例上的表现')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(case_names)
    ax.legend(loc='upper left', ncol=3)
    ax.set_ylim(0, 1.1)

    # 添加数值标签
    for i, method in enumerate(methods):
        for j, value in enumerate(results[method]):
            ax.text(j + width * i, value + 0.02, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150)
    print("\n结果已保存到 'method_comparison.png'")
    plt.show()


def test_custom_weights():
    """测试自定义权重"""
    print("\n" + "=" * 60)
    print("自定义权重测试")
    print("=" * 60)

    # 创建测试文件
    midi1_path, midi2_path = create_sample_midi()

    # 定义自定义权重（强调旋律）
    custom_weights = {
        'melodic_shape': 0.5,
        'rhythmic_groove': 0.2,
        'harmonic_flavor': 0.15,
        'structural_pattern': 0.1,
        'expressive_intensity': 0.05
    }

    # 使用自定义权重
    comparator = MIDISimilarityToolkit(method='style', weights=custom_weights)
    similarity = comparator.compare(midi1_path, midi2_path)

    print(f"使用自定义权重的相似度: {similarity:.4f}")
    print("自定义权重:")
    for k, v in custom_weights.items():
        print(f"  {k}: {v:.2f}")

    # 与默认权重比较
    default_comparator = MIDISimilarityToolkit(method='style')
    default_similarity = default_comparator.compare(midi1_path, midi2_path)

    print(f"\n默认权重相似度: {default_similarity:.4f}")
    print(f"差异: {abs(similarity - default_similarity):.4f}")

    # 清理
    try:
        os.remove(midi1_path)
        os.remove(midi2_path)
    except:
        pass


def test_cache_performance():
    """测试缓存性能"""
    print("\n" + "=" * 60)
    print("缓存性能测试")
    print("=" * 60)

    # 创建测试文件
    midi1_path, midi2_path = create_sample_midi()

    comparator = MIDISimilarityToolkit(method='balanced')

    # 第一次比较（无缓存）
    print("第一次比较（无缓存）:")
    start_time = time.time()
    similarity1 = comparator.compare(midi1_path, midi2_path, use_cache=False)
    time1 = time.time() - start_time
    print(f"  相似度: {similarity1:.4f}, 耗时: {time1:.4f}秒")

    # 第二次比较（有缓存）
    print("第二次比较（有缓存）:")
    start_time = time.time()
    similarity2 = comparator.compare(midi1_path, midi2_path, use_cache=True)
    time2 = time.time() - start_time
    print(f"  相似度: {similarity2:.4f}, 耗时: {time2:.4f}秒")

    # 第三次比较（使用缓存）
    print("第三次比较（使用缓存）:")
    start_time = time.time()
    similarity3 = comparator.compare(midi1_path, midi2_path, use_cache=True)
    time3 = time.time() - start_time
    print(f"  相似度: {similarity3:.4f}, 耗时: {time3:.4f}秒")

    print(f"\n性能提升:")
    print(f"  第一次 vs 第二次: {time1 / time2:.2f}倍")
    print(f"  第一次 vs 第三次: {time1 / time3:.2f}倍")

    # 获取缓存信息
    cache_info = comparator.get_cache_info()
    print(f"\n缓存信息: {cache_info}")

    # 清理
    try:
        os.remove(midi1_path)
        os.remove(midi2_path)
    except:
        pass


def run_all_tests():
    """运行所有测试"""
    print("开始运行所有测试...")
    print()

    # 运行各个测试
    test_basic_comparison('out.mid', 'target.mid')
    #test_performance_scaling()
    test_method_comparison()
    test_custom_weights()
    test_cache_performance()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 检查依赖
    try:
        import pretty_midi
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install pretty_midi numpy matplotlib scipy scikit-learn")
        exit(1)

    # 运行测试
    run_all_tests()