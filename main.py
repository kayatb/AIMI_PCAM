# import dataset
from train import train
from models import EqvModel

import math

import argparse
# from json import load
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torch import nn
from copy import Error, deepcopy

def add_args():
    """ Add a lot of commande line arguments for specifying all sorts of hyperparams and other settings. """
    parser = argparse.ArgumentParser("Set PCAM model", add_help=False)

    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--no_startup_epoch", action='store_true')
    parser.add_argument("--seed", default=432, type=int)
    parser.add_argument("--load_checkpoint", action='store_true')

    parser.add_argument("--weight_decay_base", choices=['zero', 'pretrained'], default=None)
    parser.add_argument("--weight_decay_strength", type=float, default=0.001)

    # Training settings.
    parser.add_argument("--lr", default=0.0001, type=float)
    # parser.add_argument("--lr_drop", default=70, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    parser.add_argument("--output_dir", default="saved_models/", help="path where to save trained model, empty for no saving")

    parser.add_argument("--tqdm", action='store_true')
    parser.add_argument("--max_batches_per_epoch", default=None, type=int)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--file_name", type=str, default=None)

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


def load_model(model_name, pretrained, args):
    """ Return the specified model. """
    # TODO: add more models here.
    if model_name == "resnet18":
        if not pretrained:
            return torchvision.models.resnet18(num_classes=2, pretrained=False), None
        else:
            resnet = torchvision.models.resnet18(num_classes=1000, pretrained=True)
            resnet.fc = nn.Linear(resnet.fc.in_features, 2)
            return resnet, lambda model: model.fc.parameters()
    if model_name == "escnn":
        base_hidden_dims = [12, 12, None, 24, 24, None, 48, 48, None, 96, 96, None, 96, 96, None, 192, 192]
        scale = math.sqrt(args.rotations * 2)
        hidden_dims = [
            dim if dim is None else int(dim / scale)
            for dim in base_hidden_dims
        ]
        
        model = EqvModel(hidden_dims)
        return model, lambda model: model.parameters()
    else:
        raise Error(f"Unknown model name {model_name}.")


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

    model, startup_params = load_model(args.model, args.pretrained, args)

    if args.weight_decay_base == "zero":
        weight_decay_base = 0
    elif args.weight_decay_base == "pretrained":
        weight_decay_base = deepcopy(model)
    else:
        weight_decay_base = None

    best_model, logging_info = train(model, args, None if args.no_startup_epoch else startup_params, weight_decay_base, args.weight_decay_strength)

    # train, val, test = dataset.load_dataset(4)

    # dataiter = iter(train)
    # images, labels = dataiter.next()

    # # Create a grid from the images and show them
    # img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=False)
