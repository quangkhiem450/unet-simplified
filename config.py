# dataset
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
data_folder = '../../dataset_small'
batch_size = 2
num_workers = 4

# train
pretrained_path = ''
experiment_path = 'runs'
epoch_number = 10
learning_rate = 1e-4

# validation
model_path_val = 'runs/checkpoint_best.pt'

# prediction
model_path_pred = 'runs/checkpoint_best.pt'
save_folder = 'runs/predict'

