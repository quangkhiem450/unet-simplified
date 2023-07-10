import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
import torch
from tqdm import tqdm
from efficientunet import *
from dataset import BonelevelDataset
from unet import UNet
from engine import train_one_epoch, valid_one_epoch
import config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    train_dataset = BonelevelDataset(image_dir=os.path.join(config.data_folder, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_dataset = BonelevelDataset(image_dir=os.path.join(config.data_folder, 'valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=4)
    model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True).to(device)
    # model = UNet(dimensions = 1).to(device)
    if not os.path.isdir(config.experiment_path):
        os.mkdir(config.experiment_path)
    
    if os.path.isfile(config.pretrained_path):
        model.load_state_dict(torch.load(config.pretrained_path, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # train and validation the model base on the number of epoch
    best_loss = 100000
    for epoch in range(config.epoch_number):
        train_loss, train_dice = train_one_epoch(model, train_loader, train_dataset, 
                                                 optimizer, criterion, device, epoch, config.epoch_number)
        valid_loss, valid_dice = valid_one_epoch(model, valid_loader, valid_dataset, 
                                                 criterion, device, epoch, config.epoch_number)
        print('Epoch: {} - Train Loss: {:.6f} - Train Dice: {:.6f} - Valid Loss: {:.6f} - Valid Dice: {:.6f}'
              .format(epoch + 1, train_loss, train_dice, valid_loss, valid_dice))

        torch.save(model.state_dict(), os.path.join(config.experiment_path, 'checkpoint_{}.pt'.format(epoch)))
        print('Model saved at epoch {}'.format(epoch))
        
        if valid_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(config.experiment_path, 'checkpoint_best.pt'))
            best_loss = valid_loss
            print('Model best saved at epoch {} - loss {:.6f} - dice {:.6f}'.format(epoch, best_loss, valid_dice))
            
        print('-' * 100)
    
    torch.save(model.state_dict(), os.path.join(config.experiment_path, 'checkpointl_last.pt'))
    print('Model saved at the end of training')

if __name__ == "__main__":
    main()
