from datasets import Dataset


def process_example(example):
        """处理单个样本"""
        result = {}
        
        # 保持 audio_input 不变
        if 'audio_input' in example:
            result['audio_input'] = example['audio_input']
        
        # 展开 context_notes
        if 'context_notes' in example:
            context_data = example['context_notes']
            result['context_duration'] = context_data.get('duration', [])
            result['context_fret'] = context_data.get('fret', [])
            result['context_technique'] = context_data.get('technique', [])
        
        # 展开 target_notes
        if 'target_notes' in example:
            target = example['target_notes']
            result['target_duration'] = target.get('duration', [])
            result['target_fret'] = target.get('fret', [])
            result['target_technique'] = target.get('technique', [])
        
        return result
    
out_path = "output/Model/dataset/fixed"
dataset_path = "output/Model/dataset"
dataset = Dataset.load_from_disk(dataset_path).map(process_example, batched=True, writer_batch_size = 100)
dataset.save_to_disk(out_path)