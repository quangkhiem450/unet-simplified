import torch

# this file contains all the implementations of the segmentation metrics

# implementation of dice score 
def dice_score(pred, target, smooth=0.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    iflat = pred.contiguous().view(-1).float()
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# implementation of accuracy score
def accuracy_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / (correct.numel())
    return accuracy

# implementation of precision score
def precision_score(pred, target, smooth=0.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision

# implementation of recall score (độ nhạy)
def recall_score(pred, target, smooth=0.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

# implementation of specification score (độ đặc hiệu)
def specificity_score(pred, target, smooth=0.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    specificity = (tn + smooth) / (tn + fp + smooth)
    return specificity

# implementation of Jaccard score (iou)
def jaccard_score(pred, target, smooth=0.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)