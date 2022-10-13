""" 
General functions to calculate the average metrics (area size, precision, IAUC, DAUC, infidelity, sensitivity) 
over the entire validation set. 

------------------------------------------------------------------------------------------------------------------------

Code was partially taken and adapted from the following paper:

Li, L., Wang, B., Verma, M., Nakashima, Y., Kawasaki, R., & Nagahara, H. (2021). 
SCOUTER: Slot attention-based classifier for explainable image recognition. 
In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1046-1055).

Code available at: https://github.com/wbw520/scouter
Commit: 5885b82 on Sep 7, 2021

"""

from __future__ import print_function

import argparse

# import json
import os
import os.path

import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# from dataset.ConText import ConText, MakeListImage
from sloter.slot_model import SlotModel
from train import get_args_parser
from torchvision import transforms
from torchvision.datasets import PCAM
from dataset.transform_func import make_transform
from sloter.utils.vis import apply_colormap_on_image


# from metrics.utils import exp_data
# from metrics.area_size import calc_area_size
# from metrics.IAUC_DAUC import calc_iauc_and_dauc_batch
# from metrics.precision import calc_precision
# from metrics.saliency_evaluation.eval_infid_sen import calc_infid_and_sens


def calc_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    """
    # Get the labels of the predictions by taking the index of the max value
    # for each row in the predictions matrix.
    preds = np.argmax(predictions.cpu().numpy(), axis=1)
    accuracy = np.sum(preds == targets.cpu().numpy())
    accuracy /= len(targets)

    return accuracy


def evaluate_model(model, data_loader, device):
    """
    Performs the evaluation of the PCAM model on a given dataset.

    Args:
      model: An instance of the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.
    """
    accuracies = []

    model.eval()  # Put model in evaluation mode.

    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            labels = labels.to(device)

            predictions = model(x).squeeze(dim=1)
            accuracies.append(calc_accuracy(predictions, labels))

    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_accuracy


def prepare(batch_size):
    """Prepare model, datasets etc. for evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser("model training and evaluation script", parents=[get_args_parser()])
    parser.add_argument("--model_name", required=True, help="Filename of saved model")
    parser.add_argument(
        "--csv",
        default="data/imagenet/LOC_val_solution.csv",
        type=str,
        help="Location of the CSV file that contains the bounding boxes",
    )

    parser.add_argument(
        "--area_prec", action="store_true", help="Whether to calculate the area size and precision metrics"
    )
    parser.add_argument("--auc", action="store_true", help="Whether to calculate the IAUC and DAUC metrics")
    parser.add_argument(
        "--saliency", action="store_true", help="Whether to calculate the infidelity and sensitivity metrics"
    )

    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ["num_classes", "lambda_value", "power", "slots_per_class"]
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    # model_name = (
    #     f"{args.dataset}_"
    #     + f"{'use_slot_' if args.use_slot else 'no_slot_'}"
    #     + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"
    #     + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}"
    #     + "checkpoint.pth"
    # )

    args.use_pre = False

    device = torch.device(args.device)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # Retrieve the data. We only need to evaluate the validation set.
    dataset_val = PCAM("data", "test", transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # Load the model from checkpoint.
    model = SlotModel(args)
    checkpoint = torch.load(f"{args.output_dir}" + args.model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    return model, data_loader_val, transform, device, args


def prepare_data_point(data, transform, device):
    """Prepare a single datapoint (image, label, file name) to be used in evaluation or
    explanation generation."""
    image = data[0][0]
    label = data[1][0].item()
    # fname = os.path.basename(data["names"][0])[:-5]  # Remove .JPEG extension.

    image_orl = Image.fromarray(
        (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
        mode="RGB",
    )

    image = transform(image_orl)
    transform2 = transforms.Compose([transforms.Normalize([0.7008, 0.5384, 0.6916], [0.2350, 0.2774, 0.2128])])
    norm_image = transform2(image)
    norm_image = norm_image.to(device, dtype=torch.float32)

    return norm_image, label, image


def generate_explanations(model, data_loader_val, device, model_name, save_path=None):
    """Generate and save the explanations of the model for the images in the validation set.
    Only the ground truth explanation and the least similar class are generated/saved."""
    if not save_path:
        os.makedirs(f"exps/{model_name}/positive", exist_ok=True)
        os.makedirs(f"exps/{model_name}/negative", exist_ok=True)
        save_path = f"exps/{model_name}"

    # for i, data in enumerate(data_loader_val):
    for i, data in enumerate(data_loader_val):
        print(i)
        image, label, unnorm_img = prepare_data_point(data, transform, device)
        # Explanation image is generated during forward pass of image in the model.
        _ = model(torch.unsqueeze(image, dim=0), save_id=(label, abs(label - 1), save_path, i))
        save_explanation(unnorm_img, f"{save_path}/positive", i, label)
        # save_explanation(unnorm_img, f"{save_path}/negative", i)


def save_explanation(image, dir, exp_id, label):
    # image = image / 2 + 0.5     # unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.savefig("raw_image.png", bbox_inches="tight", pad_inches=0.0)
    raw_image = Image.open("raw_image.png")

    slot_image = np.array(
        Image.open(f"{dir}/{exp_id}.png").resize(raw_image.size, resample=Image.BILINEAR), dtype=np.uint8
    )

    heatmap_only, heatmap_on_image = apply_colormap_on_image(raw_image, slot_image, "jet")
    heatmap_on_image.save(f"{dir}/img_{exp_id}_l{label}.png")


if __name__ == "__main__":
    # Use batch size = 1 to handle a single image at a time.
    # model, data_loader_val, transform, device, args = prepare(128)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # acc = evaluate_model(model, data_loader_val, device)
    # print("Test set accuracy:", acc)

    model, data_loader_val, transform, device, args = prepare(1)
    # Generate all explanation images.
    generate_explanations(model, data_loader_val, device, args.model_name)
