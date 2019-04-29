# local imports
from reconstruction.unet.general_unet import GeneralUnet
from data import ReconstructionDataset
from utils.utils import to_pil_img, save_checkpoint, create_unet
from utils.utils import train, predict, test
from utils.argsparser import ArgsParserRecon

# library imports
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold= 128*128+1)

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    down_layers, up_layers = create_unet(args.cfg_file)
    print(down_layers, up_layers)
    unet = GeneralUnet(down_layers, up_layers)
    unet = unet.to(device)

    input_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()])

    label_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()])

    train_dataset = ReconstructionDataset.ReconstructionDataset(args.train_csv, args.train_input_dir, args.train_gt_dir, input_transforms=input_transform, label_transforms= label_transform)
    val_dataset = ReconstructionDataset.ReconstructionDataset(args.val_csv, args.val_input_dir, args.val_gt_dir, input_transforms=input_transform, label_transforms= label_transform)

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

    # optimizer = torch.optim.Adam(unet.parameters(), lr=0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = torch.optim.SGD(unet.parameters(), lr= .05, momentum=.9, weight_decay= 0)
    criterion = torch.nn.MSELoss().to(device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35])

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(unet, optimizer, criterion, device, train_loader, epoch, args.log_interval)
        test_loss = test(unet, device, val_loader, epoch, args.log_interval)

        if test_loss < best_loss:
            best_loss = test_loss
            is_best = True

        # save the model every epoch
        save_checkpoint({
            'epoch' : epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss' : test_loss
            }, is_best)

        is_best = False

if __name__ == '__main__':
    args = ArgsParserRecon().parse_args()
    if args.val_input_dir is None:
        args.val_input_dir = args.train_input_dir
        args.val_gt_dir = args.train_gt_dir
    if args.checkpoint:
        predict(args.checkpoint, args)
    else:
        main(args)

# python main.py --resize 572 --train-csv ./data/train_data.csv --val-csv ./data/val_data.csv --train-input-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/train2017 --train-gt-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/train2017_gt --val-input-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/val2017 --val-gt-dir /scratch/kingspeak/serial/u0853593/images/reconstruction/val2017_gt --log-interval 500
# python run_reconstruction.py --resize 128 --train-csv data/csv_files/mnist_train.csv --val-csv data/csv_files/mnist_val.csv --train-input-dir data/tpmnist_avg --train-gt-dir data/mnist_gt --log-interval 500 --cfg-file reconstruction/models/model1.cfg --checkpoint model_best_5.pth
# python main.py --resize 572 --pred-img ./tpmnist_avg/0/13238.jpg --checkpoint ./model_best_1.pth
# python run_reconstruction.py --resize 128 --train-csv data/csv_files/cifar_recon_train.csv --val-csv data/csv_files/cifar_recon_val.csv --train-input-dir data/tp_cifar_avg --train-gt-dir data/cifar --log-interval 500 --cfg-file reconstruction/models/model4.cfg --checkpoint model4_cifar_best_20.pth

