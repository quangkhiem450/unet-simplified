import torch
from torch import nn
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from metrics import dice_score, accuracy_score, precision_score, recall_score, specificity_score, jaccard_score
import config 


# create function to train one epoch of Unet model
def train_one_epoch(model, dataloader, dataset, optimizer, criterion, device, epoch, epoch_number):
    model.train()
    epoch_loss, dices = 0, 0
    with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epoch_number}', unit='img') as pbar:
        for i, batch in enumerate(dataloader):
            input = batch['image']
            target = batch['mask']
            input = input.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            dices += dice_score(output, target)
            
            pbar.set_postfix(**{'loss (batch)': loss.item(), 'dice (batch)': float(dice_score(output, target))})
            pbar.update(input.shape[0])
            
    return epoch_loss / len(dataloader), dices / len(dataloader)


def valid_one_epoch(model, dataloader, dataset, criterion, device, epoch, epoch_number):
    model.eval()
    epoch_loss, dices = 0, 0
    with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epoch_number}', unit='img') as pbar:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input = batch['image']
                target = batch['mask']
                input = input.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                
                output = model(input)
                loss = criterion(output, target)
                epoch_loss += loss.item()
                dices += dice_score(output, target)
                
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'dice (batch)': float(dice_score(output, target))})
                pbar.update(input.shape[0])
                
    return epoch_loss / len(dataloader), dices / len(dataloader)


# validation model on validation set and show all metrics
def validation(model, dataloader, dataset, criterion, device, epoch, epoch_number):
    model.eval()
    epoch_loss, dices, accuracies, precisions, recalls, specificities, jaccards = 0, 0, 0, 0, 0, 0, 0
    with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epoch_number}', unit='img') as pbar:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input = batch['image']
                target = batch['mask']
                input = input.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)

                output = model(input)
                loss = criterion(output, target)
                epoch_loss += loss.item()
                dices += dice_score(output, target)
                accuracies += accuracy_score(output, target)
                precisions += precision_score(output, target)
                recalls += recall_score(output, target)
                specificities += specificity_score(output, target)
                jaccards += jaccard_score(output, target)

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'dice (batch)': float(dice_score(output, target))})
                pbar.update(input.shape[0])

    return epoch_loss / len(dataloader), dices / len(dataloader), accuracies / len(dataloader), precisions / len(
        dataloader), recalls / len(dataloader), specificities / len(dataloader), jaccards / len(dataloader)

# predict the single image and return the predicted mask
def predict(model, image_path, save_folder, device):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ori_height, ori_width = image.shape
    
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])
    sample = transform(image)
    
    sample = sample.unsqueeze(0)
    sample = sample.to(device, dtype=torch.float32)
    output = model(sample)
    output = output.squeeze(0)
    output = output.cpu().detach().numpy()
    output = np.argmax(output, axis=0)
    output = output.astype(np.uint8)
    
    output = cv2.resize(output, (ori_width, ori_height))
    output = Image.fromarray(output)
    output.save(os.path.join(save_folder, os.path.basename(image_path)))
