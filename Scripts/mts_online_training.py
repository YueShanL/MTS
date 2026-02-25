import datasets

from model.mts_config import MTSGenConfig
from model.mts_generate import MTSGen
from model.trainer import train_mixed_model

linux = 1
debug = 0
if __name__ == '__main__':
    output_path = "output/Model/tf_online" if linux else "../output/Model/test"

    training_size = 507000
    eval_size = 800
    batch_size = 16

    dataset = datasets.load_dataset('astune/mts_dataset', streaming=True, split='train')
    val_dataset = dataset.take(eval_size)
    train_dataset = dataset.skip(eval_size).take(training_size)

    config = MTSGenConfig.mtsGen_150m()
    model = MTSGen(config)
    # model.load_state_dict(torch.load(f'Scripts/final_model.pth'))
    model.to('cuda')
    train_mixed_model(model, train_dataset, val_dataset=val_dataset,
                      num_epochs=100, batch_size=batch_size, output_path=output_path,
                      scheduler_type="teacher_forced", epoches_len=training_size//batch_size + 1)