import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def compute_IoU(cm):

    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou[1:])

def eval_net_loader(net, val_loader, n_classes, device = 'cpu'):
    net.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes,n_classes))

    for i, batch in enumerate(tqdm(val_loader)):
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = net(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for j in range(len(masks)):
            true = masks[j].cpu().detach().numpy().flatten()
            pred = preds[j].cpu().detach().numpy().flatten()
            cm += confusion_matrix(true, pred, labels=labels)

    class_iou, mean_iou = compute_IoU(cm)
    return class_iou, mean_iou