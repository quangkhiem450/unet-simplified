import torch
import os
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine import predict
from dataset import BonelevelDataset
from unet import UNet
import config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = UNet(dimensions=1).to(device)
    model.load_state_dict(torch.load(config.model_path_pred, map_location=device))
    
    if not os.path.isdir(config.save_folder):
        os.mkdir(config.save_folder)
    
    image_dir = os.path.join(config.data_folder, 'test')
    image_path_list = [y for x in os.walk(image_dir) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    
    for image_path in tqdm(image_path_list):
        predict(model, image_path, config.save_folder, device)


if __name__ == "__main__":
    main()