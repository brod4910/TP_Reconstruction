# local imports
from reconstruction.unet.unet_sections import *

# python imports
import os
import shutil
import configparser

# library imports
from sklearn.model_selection import KFold
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

def kfold_to_csv(folder_path, dest_folder, n_splits= 10):
    path = os.path.join(os.getcwd(), folder_path)
    dirs = [os.path.join(path, directory) for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
    kf = KFold(n_splits= n_splits, shuffle= True, random_state= 1)
    kfolds = []
    list.sort(dirs)
    print(dirs)
    for directory in dirs:
        img_names = np.array([name for name in os.listdir(directory)])
        kfolds.append(kf.split(img_names))

    train_imgs = []
    val_imgs = []
    for fold, directory in zip(kfolds, dirs):
        img_names = np.array([name for name in os.listdir(directory)])
        for train_idx, val_idx in fold:
            train_imgs.append(img_names[train_idx])
            val_imgs.append(img_names[val_idx])
            break

    with open(os.path.join(dest_folder, 'mnist_train.csv'), 'w') as f:
        for i, n_class in enumerate(train_imgs):
            for img in n_class:
                f.write('{}/{}\n'.format(i, img))

    with open(os.path.join(dest_folder, 'mnist_val.csv'), 'w') as f:
        for i, n_class in enumerate(val_imgs):
            for img in n_class:
                f.write('{}/{}\n'.format(i, img))

def train(model, optimizer, criterion, device, train_loader, epoch, log_interval):
    model.train()
    optimizer.zero_grad()
    total_train_loss = 0

    total_train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data['input'], data['label']
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
                100. * batch_idx / len(train_loader.dataset), loss.item()), flush= True)
            print('outputs:', output, flush= True)

        del inputs, targets, loss, output

    print('\nAveraged loss for training epoch: {:.6f}'.format(total_train_loss/len(train_loader.dataset)))

def test(model, device, val_loader, epoch, log_interval, loss_fn= 'mse'):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, targets = data['input'], data['label']
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            if loss_fn == 'mse':
                test_loss += F.mse_loss(output, targets).item()
            elif loss_fn == 'ce':
                test_loss += F.cross_entropy(output, targets).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()

            if (batch_idx + 1) % log_interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(val_loader.dataset),
                100. * batch_idx / len(val_loader.dataset), test_loss) , flush= True)
                print('outputs:', output, flush= True)

            del inputs, targets, output
    print('\nTest set: Average loss: {:.6f}\tCorrect: {}/{} ({:.6f})\n'.format(test_loss/batch_idx+1, correct, 
                len(val_loader.dataset), correct/len(val_loader.dataset)), flush= True)

    return test_loss


def predict(checkpoint_file, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # print(checkpoint)

    unet = UNet(3, 1)
    unet = unet.to(device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()

    transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor()])

    if args.pred_img:
        img = Image.open(args.pred_img)
        img = transform(img)
        img = img.expand(1, -1, -1, -1)

        output = unet(img)
        img = to_pil_img(output)
        plt.imshow(img)
        plt.show()
    else:
        val_dataset = TransparentDataset.TransparentDataset(args.val_csv, args.val_input_dir, args.val_gt_dir, 500, input_transforms=transform, pred= 'yes')

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size= args.batch_size, 
            shuffle= True, 
            num_workers= 2,
            pin_memory= True
            )
        with torch.no_grad():
            for batch_idx, (inputs, img_name) in enumerate(val_loader):
                output = unet(inputs)
                img = to_pil_img(output)

                if batch_idx == 0:
                    image = plt.imshow(img)
                else:
                    image.set_data(img)

                plt.draw()
                plt.pause(2)
                print(batch_idx, img_name)

def create_model(cfg):
    config = configparser.ConfigParser()
    config.read(cfg)

    feature_layers = []
    linear_layers = []
    for section in config.sections():
        kwargs = {}
        if 'convolution' in section:
            conv = config[section]
            for param in conv:
                if param not in ['activation', 'batch_norm']:
                    kwargs[param] = int(conv[param])
                elif param == 'activation':
                    activation = torch.nn.ReLU(inplace= True)
                elif param == 'batch_norm':
                    batch_norm = conv[param]
                    if batch_norm == '1':
                        bnorm = torch.nn.BatchNorm2d(kwargs['out_channels'])
            conv_layer = torch.nn.Conv2d(**kwargs)
            feature_layers += [conv_layer, bnorm, activation]
        elif 'maxpool' in section:
            pool = config[section]
            for param in pool:
                kwargs[param] = int(pool[param])

            feature_layers += [torch.nn.MaxPool2d(**kwargs)]
        elif 'linear' in section:
            linear = config[section]
            # TODO: change from eval to another hand-written function
            for param in linear:
                kwargs[param] = eval(linear[param])

            linear_layers += [torch.nn.Linear(**kwargs)]
    return torch.nn.Sequential(*feature_layers), torch.nn.Sequential(*linear_layers)

def create_unet(cfg):
    config = configparser.ConfigParser()
    config.read(cfg)
    curr_section = None
    down_layers = []
    up_layers = []

    for section in config.sections():
        kwargs = {}
        if section == 'down_layers':
            curr_section = section
            continue
        elif section == 'up_layers':
            curr_section = section
            continue

        if curr_section == 'down_layers':
            if 'down' in section:
                down = config[section]
                for param in down:
                    kwargs[param] = int(down[param])
                print(kwargs)
                down_layers += [DoubleConv(**kwargs)]
            elif 'maxpool' in section:
                pool = config[section]
                for param in pool:
                    kwargs[param] = int(pool[param])
                down_layers += [MaxPool(**kwargs)]
        elif curr_section == 'up_layers':
            if 'down' in section:
                down = config[section]
                for param in down:
                    kwargs[param] = int(down[param])
                up_layers += [DoubleConv(**kwargs)]
            elif 'up' in section:
                up = config[section]
                for param in up:
                    kwargs[param] = int(up[param])
                up_layers += [TransposeConv(**kwargs)]
            elif 'singleconv' in section:
                single = config[section]
                for param in single:
                    try:
                        kwargs[param] = int(single[param])
                    except Exception as e:
                        kwargs[param] = single[param]
                up_layers += [SingleConv(**kwargs)]

    return down_layers, up_layers

def to_pil_img(tensor):
    pil = transforms.ToPILImage()
    img = pil(tensor.squeeze())
    return img

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

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth'.format(state['epoch']))