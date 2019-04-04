# local imports
from reconstruction.unet.unet_model import UNet
from data import ClassificationDataset
from utils.utils import to_pil_img, save_checkpoint
from utils.utils import train, predict, test
from utils.argsparser import ArgsParserClass
from classification.model import Model

# python imports
import shutil

# library imports
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import numpy as np

def main(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Model(args.cfg_file).to(device)
    # model = models.resnet18().to(device)

    data_transforms = transforms.Compose(
        [transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()]
        )

    train_dataset = ClassificationDataset.ClassificationDataset(args.train_csv, 
        args.root_dir, transform= data_transforms)
    val_dataset = ClassificationDataset.ClassificationDataset(args.val_csv, 
        args.root_dir, transform= data_transforms)

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

    optimizer = torch.optim.SGD(model.parameters(), lr= args.lr, momentum= .9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(model, optimizer, criterion, device, train_loader, epoch, args.log_interval)
        test_loss = test(model, device, val_loader, epoch, args.log_interval, loss_fn= 'ce')


if __name__ == '__main__':
    args = ArgsParserClass().parse_args()
    main(args)

# python run_classification.py --lr .001 --resize 224 --cfg-file classification/models/model2.cfg --root-dir data/tp_cifar_avg --train-csv data/csv_files/cifar_train.csv --val-csv data/csv_files/cifar_test.csv