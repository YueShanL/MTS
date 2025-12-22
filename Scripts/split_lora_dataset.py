from datasets import DatasetDict
from sklearn.model_selection import train_test_split

# 假设你现有的 dataset_dict 结构
# dataset_dict = DatasetDict({'train': train_dataset})
debug = 0
linux = 1
dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset"
dataset_out_path = "output/Lora/dataset/splited" if linux else "../output/Lora/dataset"
# 获取原始训练集
train_dataset = DatasetDict.load_from_disk(dataset_path)['train']

# 分割数据集 (80%训练, 20%评估)
train_val_split = train_dataset.train_test_split(
    test_size=0.2,  # 评估集比例
    seed=42,        # 随机种子
    shuffle=True    # 是否打乱
)

# 创建新的 DatasetDict
new_dataset_dict = DatasetDict({
    'train': train_val_split['train'],
    'eval': train_val_split['test']  # 或者 'validation'
})

new_dataset_dict.save_to_disk(dataset_out_path)

print(f"原始训练集大小: {len(train_dataset)}")
print(f"新训练集大小: {len(new_dataset_dict['train'])}")
print(f"评估集大小: {len(new_dataset_dict['eval'])}")