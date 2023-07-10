import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import dice_score, accuracy_score, precision_score, recall_score, specificity_score, jaccard_score
from engine import validation
from dataset import BonelevelDataset
from unet import UNet
import config


# validation model by metrics based on validation set
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    valid_dataset = BonelevelDataset(image_dir=os.path.join(config.data_folder, 'valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    model = UNet(dimensions=1).to(device)
    model.load_state_dict(torch.load(config.model_path_val, map_location=device))
    criterion = torch.nn.BCEWithLogitsLoss()
    
    loss, dice, acc, prec, rec, spec, jacc = validation(model, valid_loader, valid_dataset, 
                                                        criterion, device, 0, 1)
    
    # print all the information of metrics
    print("Validation Loss: {:.6f}". format(loss))
    print("Validation Dice: {:.6f}". format(dice))
    print("Validation Accuracy: {:.6f}". format(acc))
    print("Validation Precision: {:.6f}". format(prec))
    print("Validation Recall: {:.6f}". format(rec))
    print("Validation Specificity: {:.6f}". format(spec))
    print("Validation Jaccard: {:.6f}". format(jacc))    

if __name__ == "__main__":
    main()
