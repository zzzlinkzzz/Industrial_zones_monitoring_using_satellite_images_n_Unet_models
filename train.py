import torch
import torch.nn as nn
from torch import optim

from Unet_model import Unet_model

from loader import make_dataloaders
from validation import eval_net_loader

import albumentations as A
from tqdm import tqdm
from optparse import OptionParser
import os,sys
# ============================================================
accumulation_steps = 16
gamma=0.97
dir_data = './data_512_128'
dir_checkpoint = './cp_unet_512_v5_cumulated/'
num_workers =  1

train_transform = A.Compose(
        [
            A.Rotate(limit=90, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    )
is_culmulated = True
# ============================================================
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='eta', default=0.1,
                      type='float', help='learning rate')
    (options, args) = parser.parse_args()
    return options
# ============================================================
def train_epoch(train_loader, criterion, optimizer):
    net.train()
    epoch_loss = 0
    batch_num = len(train_loader)

    for i, batch in enumerate(tqdm(train_loader)):
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = net(imgs)
        # probs = torch.softmax(outputs, dim=1)

        loss = criterion(outputs, masks)
        epoch_loss += loss.detach().to('cpu').item()
        loss.backward()

        if is_culmulated:
            loss /= accumulation_steps
            if ((i+1) % accumulation_steps == 0) or ((i+1) == batch_num):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
    
    print(f'Loss: {epoch_loss/i:.2f}')
# ============================================================
def validate_epoch(val_loader, device):
    classIOU, meanIOU = eval_net_loader(net, val_loader, 5, device)
    print('Class IoU:', ','.join(f'{x:.3f}' for x in classIOU))
    print(f'Mean IoU: {meanIOU:.3f}') #  |  
    return meanIOU
# ============================================================
def train_net(train_loader, val_loader, net, device, epochs = 1, batch_size = 2, eta=0.1, save_cp=True):
    print(f'''
Training params:
    Epochs: {epochs}
    Batch_size: {batch_size}
    Learning rate: {eta}
    Training size: {len(train_loader.dataset)}
    Validation size: {len(val_loader.dataset)}
    Device: {device}
    accumulation steps: {accumulation_steps}
    Decreasing Learning rate by {round((1-gamma),2)*100}% per epoch
    ''')

    optimizer = optim.Adam(net.parameters(),  lr= eta, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    # criterion = nn.CrossEntropyLoss(ignore_index = 0, 
    #                                 weight=torch.as_tensor(
    #                                     [0., 1., 1., 5., 1., 1.]).cuda())
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    best_precision = 0

    cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
    
    for epoch in range(len(cp_list),epochs):
        print(f'''============================================================\nStart epoch {epoch+1}\nTraining session''')
        cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
        
        if len(cp_list) != 0:
            trained_model = f'CP{len(cp_list)}.pth'
            # net = Unet_model(in_channels=3 ,n_classes=5)
            net.load_state_dict(torch.load(dir_checkpoint+trained_model))
            net.eval()
            print('load '+ trained_model)

        train_epoch(train_loader, criterion, optimizer)
        print("Validation session")
        precision = validate_epoch(val_loader, device)
        # scheduler.step()
        
        if save_cp and (precision>best_precision):
            state_dict = net.state_dict()
            best_precision = precision

        torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
        print('Checkpoint {} saved !'.format(epoch + 1))
# ============================================================
if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = get_args()
    
    train_loader, val_loader = make_dataloaders(dir_data, train_transform, None, args.batch_size, num_workers)
    
    net = Unet_model(in_channels=3, n_classes=5)

    net.to(device)

    try:
        train_net(train_loader, val_loader, net, device, epochs=args.epochs, batch_size=args.batch_size, eta=args.eta)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)