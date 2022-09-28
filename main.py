# import dataset
from train import train

import argparse
# from json import load
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def add_args():
    """ Add a lot of commande line arguments for specifying all sorts of hyperparams and other settings. """
    parser = argparse.ArgumentParser("Set PCAM model", add_help=False)

    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--seed", default=432, type=int)
    parser.add_argument("--tqdm", action='store_true')

    # Training settings.
    parser.add_argument("--lr", default=0.0001, type=float)
    # parser.add_argument("--lr_drop", default=70, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    parser.add_argument("--output_dir", default="saved_models/", help="path where to save trained model, empty for no saving")

    # fine-tuning settings (not implemented)
    # parser.add_argument("--freeze_layers", default=2, type=int, help="number of freeze layers")
    # parser.add_argument("--pre_dir", default="pre_model/", help="path of pre-train model")
    # parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    # parser.add_argument("--resume", default=False, type=str2bool, help="resume from checkpoint")

    # data/machine set
    # parser.add_argument(
    #     "--dataset_dir", default="../PAN/bird_200/CUB_200_2011/CUB_200_2011/", help="path for save data"
    # )

    return parser


def load_model(model_name, pretrained=False):
    """ Return the specified model. """
    # TODO: add more models here.
    if model_name == "resnet18":
        return torchvision.models.resnet18(num_classes=2, pretrained=pretrained)
    else:
        raise f"Unknown model name {model_name}."


def matplotlib_imshow(img, one_channel=False):
    """ Helper function for inline image display. """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PCAM model training and evaluation script", parents=[add_args()])
    args = parser.parse_args()

    model = load_model(args.model)
    best_model, logging_info = train(model, args)

    # train, val, test = dataset.load_dataset(4)

    # dataiter = iter(train)
    # images, labels = dataiter.next()

    # # Create a grid from the images and show them
    # img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=False)