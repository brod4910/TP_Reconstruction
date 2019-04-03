import os
from sklearn.model_selection import KFold
import numpy as np

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