""" Code for generating explanations images over a whole dataset. """

from __future__ import print_function

import argparse

import os
import os.path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from sloter.slot_model import SlotModel
from train import get_args_parser
from torchvision import transforms
from torchvision.datasets import PCAM
from sloter.utils.vis import apply_colormap_on_image


def prepare(batch_size):
    """Prepare model, datasets etc. for evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser("model training and evaluation script", parents=[get_args_parser()])
    parser.add_argument("--model_name", required=True, help="Filename of saved model")

    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ["num_classes", "lambda_value", "power", "slots_per_class"]
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    args.use_pre = False

    device = torch.device(args.device)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    dataset_test = PCAM("data", "test", transform)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # Load the model from checkpoint.
    model = SlotModel(args)
    checkpoint = torch.load(f"{args.output_dir}" + args.model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    return model, data_loader_test, transform, device, args


def prepare_data_point(data, transform, device):
    """Prepare a single datapoint (image, label, file name) to be used in evaluation or
    explanation generation."""
    image = data[0][0]
    label = data[1][0].item()

    image_orl = Image.fromarray(
        (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
        mode="RGB",
    )

    image = transform(image_orl)
    transform2 = transforms.Compose([transforms.Normalize([0.7008, 0.5384, 0.6916], [0.2350, 0.2774, 0.2128])])
    norm_image = transform2(image)
    norm_image = norm_image.to(device, dtype=torch.float32)

    return norm_image, label, image


def generate_explanations(model, data_loader, device, model_name, save_path=None):
    """Generate and save the explanations of the model for the images in the dataset.
    Only the ground truth explanation and the least similar class are generated/saved."""
    if not save_path:
        os.makedirs(f"exps/{model_name}/positive", exist_ok=True)
        os.makedirs(f"exps/{model_name}/negative", exist_ok=True)
        save_path = f"exps/{model_name}"

    for i, data in enumerate(data_loader):
        image, label, unnorm_img = prepare_data_point(data, transform, device)
        # Explanation image is generated during forward pass of image in the model.
        pred = model(torch.unsqueeze(image, dim=0), save_id=(label, abs(label - 1), save_path, i))
        pred = torch.argmax(pred).item()
        save_explanation(unnorm_img, f"{save_path}/positive", i, label, pred)
        save_explanation(unnorm_img, f"{save_path}/negative", i, label, pred)

        # We don't need to generate explanations for the whole dataset.
        if i == 100:
            break


def save_explanation(image, dir, exp_id, label, pred):
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.savefig("raw_image.png", bbox_inches="tight", pad_inches=0.0)
    raw_image = Image.open("raw_image.png")

    slot_image = np.array(
        Image.open(f"{dir}/{exp_id}.png").resize(raw_image.size, resample=Image.BILINEAR), dtype=np.uint8
    )

    heatmap_only, heatmap_on_image = apply_colormap_on_image(raw_image, slot_image, "jet")
    heatmap_on_image.save(f"{dir}/img_{exp_id}_l{label}_p{pred}.png")


if __name__ == "__main__":
    # Use batch size = 1 to handle a single image at a time.
    model, data_loader, transform, device, args = prepare(1)
    # Generate all explanation images.
    generate_explanations(model, data_loader, device, args.model_name)
