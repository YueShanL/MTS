def check_lora_dataset(dataset):
    # 诊断数据集
    print("=== 数据集诊断 ===")
    print(f"数据集类型: {type(dataset)}")

    if hasattr(dataset, 'keys'):
        print(f"数据集分割: {list(dataset.keys())}")
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"  {split_name}: {len(split_data)} 个样本")

            # 检查前几个样本
            if len(split_data) > 0:
                print(f"  第一个样本的特征: {list(split_data[0].keys())}")
                for key, value in split_data[0].items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value.shape}")
                    else:
                        print(f"    {key}: {type(value)} - {str(value)[:100]}...")
            else:
                print(f"  {split_name} 分割为空!")

    # 检查 train 分割是否存在且不为空
    if 'train' not in dataset:
        print("错误: 数据集中没有 'train' 分割")
        return

    train_dataset = dataset['train']
    if len(train_dataset) == 0:
        print("错误: 训练数据集为空")
        return

    print(f"训练样本数量: {len(train_dataset)}")