# local imports
from .nn_blocks import *
from reconstruction.unet.general_unet import GeneralUnet
from data import ReconstructionDataset

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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

def kfold_to_csv(folder_path, dest_folder, file_name, n_splits= 10):
    path = os.path.join(os.getcwd(), folder_path)
    dirs = [os.path.join(path, directory) for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
    kf = KFold(n_splits= n_splits, shuffle= True, random_state= 1)
    kfolds = []
    if dirs:
        list.sort(dirs)
        for directory in dirs:
            img_names = np.array([name for name in os.listdir(directory)])
            print(len(img_names))
            kfolds.append(kf.split(img_names))
    else:
        img_names = np.array([name for name in os.listdir(path)])
        dirs = [path]
        # print(img_names)
        kfolds.append(kf.split(img_names))

    train_imgs = []
    val_imgs = []
    for fold, directory in zip(kfolds, dirs):
        img_names = np.array([name for name in os.listdir(directory)])
        for train_idx, val_idx in fold:
            print(img_names.shape, train_idx.shape)
            train_imgs.append(img_names[train_idx])
            val_imgs.append(img_names[val_idx])
            print('Number of train imgs: ', len(img_names[train_idx]))
            print('Number of val imgs: ', len(img_names[val_idx]))
            break


    with open(os.path.join(dest_folder, '{}_train.csv'.format(file_name)), 'w') as f:
        for i, n_class in enumerate(train_imgs):
            for img in n_class:
                f.write('{}\n'.format(img))

    with open(os.path.join(dest_folder, '{}_val.csv'.format(file_name)), 'w') as f:
        for i, n_class in enumerate(val_imgs):
            for img in n_class:
                f.write('{}\n'.format(img))

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

    down_layers, up_layers = create_unet(args.cfg_file)
    unet = GeneralUnet(down_layers, up_layers)
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
        img = to_pil_img(output).convert('RGB')
        plt.imshow(img)
        plt.show()
    else:
        display_imgs = []
        val_dataset = ReconstructionDataset.ReconstructionDataset(args.val_csv, args.val_input_dir, args.val_gt_dir, 500, input_transforms= transform, label_transforms= transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size= args.batch_size, 
            shuffle= True, 
            num_workers= 2,
            pin_memory= True
            )
        fig = plt.figure(figsize= (16,9))
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        fig.tight_layout()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs, target = data['input'].to(device), data['label'].to(device)
                output = unet(inputs)
                output_img = to_pil_img(output).convert('RGB')
                input_img = to_pil_img(inputs).convert('RGB')
                label_img = to_pil_img(target).convert('RGB')
                difference = difference_imgs(output_img, label_img)
                # display_imgs.append([output_img, input_img,label_img, difference])

                # if len(display_imgs) % 4 == 0:
                #     if batch_idx + 1 == 4:
                #         sub_figs = []
                #         for i, imgs in enumerate(display_imgs):
                #             sub_fig = []
                #             for j, img in enumerate(imgs):
                #                 sub_fig.append(fig.add_subplot(i + 1, 4, j + 1).imshow(img))
                #             sub_figs.append(sub_fig)
                #     else:
                #         for (imgs, sub_fig) in zip(display_imgs, sub_figs):
                #             for (img, fig) in zip(imgs, sub_fig):
                #                 fig.set_data(img)

                #     plt.draw()
                #     plt.pause(2)

                #     display_imgs = []
                            
                if batch_idx == 0:
                    sub1 = fig.add_subplot(1, 4, 1)
                    sub2 = fig.add_subplot(1, 4, 2)
                    sub3 = fig.add_subplot(1, 4, 3)
                    sub4 = fig.add_subplot(1, 4, 4)
                    remove_ticks([sub1, sub2, sub3, sub4])
                    img1 = sub1.imshow(input_img)
                    img2 = sub2.imshow(output_img)
                    img3 = sub3.imshow(label_img)
                    img4 = sub4.imshow(difference)
                else:
                    img1.set_data(input_img)
                    img2.set_data(output_img)
                    img3.set_data(label_img)
                    img4.set_data(difference)

                plt.draw()
                # plt.savefig('fig_{}'.format(batch_idx))
                plt.pause(2)
                # print(batch_idx, img_name)

def create_model(cfg):
    config = configparser.ConfigParser()
    config.read(cfg)

    feature_layers = []
    linear_layers = []
    for section in config.sections():
        if 'convolution' in section:
            kwargs = get_params(config[section])
            feature_layers += [SingleConv(**kwargs)]
        elif 'maxpool' in section:
            kwargs = get_params(config[section])
            feature_layers += [MaxPool(**kwargs)]
        elif 'residual' in section:
            kwargs = get_params(config[section])
            feature_layers += [ResidualBlock(**kwargs)]
        elif 'dense' in section:
            kwargs = get_params(config[section])
            feature_layers += [DenseBlock(**kwargs)]
        elif 'transition' in section:
            kwargs = get_params(config[section])
            feature_layers += [TransitionBlock(**kwargs)]
        elif 'linear' in section:
            kwargs = {}
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
        if section == 'down_layers':
            curr_section = section
            continue
        elif section == 'up_layers':
            curr_section = section
            continue

        kwargs = get_params(config[section])
        if curr_section == 'down_layers':
            if 'down' in section:
                down_layers += [DoubleConv(**kwargs)]
            elif 'maxpool' in section:
                down_layers += [MaxPool(**kwargs)]
            elif 'residual' in section:
                down_layers += [ResidualBlock(**kwargs)]
            elif 'preres' in section:
                down_layers += [FullPreResBlock(**kwargs)]
            elif 'dense' in section:
                down_layers += [DenseBlock(**kwargs)]
        elif curr_section == 'up_layers':
            if 'down' in section:
                up_layers += [DoubleConv(**kwargs)]
            elif 'up' in section:
                up_layers += [TransposeConv(**kwargs)]
            elif 'singleconv' in section:
                up_layers += [SingleConv(**kwargs)]
            # elif 'residual' in section:
            #     residual = config[section]
            #     kwargs = get_params(residual)

    return down_layers, up_layers

def to_pil_img(tensor, axis= 0):
    pil = transforms.ToPILImage()
    img = pil(tensor.squeeze(axis))
    return img

def to_numpy_arr(tensor, axis= 0):
    img = tensor.detach().squeeze(axis).numpy()
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

def difference_imgs(img1, img2):
    img1_cv = np.array(img1)
    img2_cv = np.array(img2)
    difference = np.abs(img2_cv - img1_cv)

    return difference

def get_params(params):
    kwargs = {}
    for param in params:
        try:
            kwargs[param] = int(params[param])
        except Exception as e:
            kwargs[param] = params[param]

    return kwargs

def remove_ticks(figs):
    if isinstance(figs, list):
        for fig in figs:
            fig.set_xticks([])
            fig.set_yticks([])
    else:
        figs.set_xticks([])
        figs.set_yticks([])
