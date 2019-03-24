# local imports
from unet.unet_model import UNet
from data import TransparentDataset

# python imports
from utils.argsparser import CreateArgsParser
import shutil
import sys

# library imports
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    unet = UNet(3, 3)
    unet = unet.to(device)

    input_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()])

    label_transform = transforms.Compose([transforms.Resize((388, 388)),
        transforms.ToTensor()])

    train_dataset = TransparentDataset.TransparentDataset(args.train_csv, args.train_input_dir, args.train_gt_dir, input_transforms=input_transform, label_transforms= label_transform)
    val_dataset = TransparentDataset.TransparentDataset(args.val_csv, args.val_input_dir, args.val_gt_dir, input_transforms=input_transform, label_transforms= label_transform)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= 6,
    pin_memory= True
    )

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= 6,
    pin_memory= True
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = torch.nn.MSELoss().to(device)

    best_loss = 0

    for epoch in range(1, args.epochs + 1):
        train(unet, optimizer, criterion, device, train_loader, epoch, args.log_interval)
        test_loss = test(unet, device, train_loader, epoch, args.log_interval)

        if test_loss < best_loss:
            best_prec1 = test_loss
            is_best = True

        # save the model every epoch
        save_checkpoint({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss' : test_loss
            }, is_best)

        is_best = False

def train(model, optimizer, criterion, device, train_loader, epoch, log_interval):

    model.train()
    optimizer.zero_grad()
    total_train_loss = 0

    total_train_loss = 0
    for batch_idx, (inputs, targets, __) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs)                     # Forward pass
        loss = criterion(output, targets)      # Compute loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader.dataset), loss.item()))
            print('outputs:', output)
            sys.stdout.flush()

        del inputs, targets, loss, output

    print('\nAveraged loss for training epoch: {:.6f}'.format(total_train_loss/len(train_loader.dataset)))


def test(model, device, val_loader, epoch, log_interval):
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, img_name) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            test_loss += F.mse_loss(output, targets).item()

            if (batch_idx + 1) % log_interval == 0:
                print('outputs:', output)
                # img = to_numpy_arr(output)
                # img = scale(img, 0, 1)
                # plt.imsave('./{}_epoch_{}.png'.format(img_name[0][:-4], epoch), img)
                sys.stdout.flush()

            del inputs, targets
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss/len(val_loader)))

    return test_loss


def to_numpy_arr(tensor):
    img = tensor.detach().numpy()
    img = img.squeeze()
    img_size = img.shape
    img = img.reshape((img_size[1], img_size[2], img_size[0]))
    return img

def scale(X, x_min, x_max):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = 1 if denom == 0 else denom
    return x_min + nom/denom 

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth'.format(epoch))

if __name__ == '__main__':
    args = CreateArgsParser().parse_args()
    main(args)

# python main.py --resize 572 --train-csv ./data/train_data.csv --val-csv ./data/val_data.csv --train-input-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/train2017 --train-gt-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/train2017_gt --val-input-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/val2017 --val-gt-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/val2017_gt --log-interval 500


