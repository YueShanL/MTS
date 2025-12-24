from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoProcessor
from training.dataset import create_musicgen_dataset
# 假设你现有的 dataset_dict 结构
# dataset_dict = DatasetDict({'train': train_dataset})
debug = 0
linux = 1
dataset_path = "output/Lora/dataset" if linux else "../output/Lora/dataset"
dataset_out_path = "output/Lora/dataset/splited" if linux else "../output/Lora/dataset"
# 获取原始训练集
train_dataset = Dataset.load_from_disk(dataset_path).shuffle().select(range(50))
data = create_musicgen_dataset(
        train_dataset,
        processor=AutoProcessor.from_pretrained('facebook/musicgen-melody'),
    )

new_data = []
progress_bar = tqdm(data, desc=f'loading')
for i in progress_bar:
    i['labels'].squeeze(0)
    new_data.append(i)
dataset = Dataset.from_list(new_data)
dataset.remove_columns('sample_rate').rename_columns(
    {'input_features':'input_audio_values',
     'labels':'target_audio_values',
     'input_ids':'text'
     })

# 分割数据集 (80%训练, 20%评估)
train_val_split = dataset.train_test_split(
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