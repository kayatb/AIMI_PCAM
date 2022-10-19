"""
Calculate the following confusion matrix metrics:
Area Under Curve, accuracy, precision, recall, F1-score and Kappa for the given model.
"""

import argparse
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train import get_args_parser
from dataset.choose_dataset import select_dataset
from sloter.slot_model import SlotModel


def invert_labels(labels):
    """Turn 0's into 1's and vice versa (necessary for SCOUTER_min models)."""
    return torch.abs(1 - labels)


def obtain_preds(model, imgs):
    model.eval()

    # Obtain the model predictions.
    with torch.no_grad():
        pred_probs = model(imgs)

    preds = torch.argmax(pred_probs, dim=1)

    return preds, pred_probs


def calc_metrics(labels, preds, pred_probs, negative, model_name):
    """Calculate all metrics."""
    if negative:
        labels = invert_labels(labels)

    # Calculate the confusion matrix.
    cm = confusion_matrix(labels.cpu(), preds.cpu())
    tn, fp, fn, tp = cm.ravel()

    # Save confusion matrix figure.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"conf_matrix_{model_name}.png", dpi=600, bbox_inches="tight")

    # Calculate the metrics.
    metrics = {}
    metrics["auc"] = roc_auc_score(labels.cpu(), pred_probs[:, 1].cpu())
    metrics["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    metrics["recall"] = tp / (tp + fn)
    metrics["precision"] = tp / (tp + fp)
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    metrics["kappa"] = cohen_kappa_score(labels.cpu(), preds.cpu())

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("model training and evaluation script", parents=[get_args_parser()])
    parser.add_argument("--model_name", required=True, help="Path to the model checkpoint")
    parser.add_argument("--negative", action="store_true", help="whether model is positive of negative SCOUTER")
    args = parser.parse_args()

    assert args.dataset == "PCAM", "Calculating the metrics is only implemented for the PCAM dataset."

    args_dict = vars(args)
    args_for_evaluation = ["num_classes", "lambda_value", "power", "slots_per_class"]
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    device = torch.device(args.device)

    # Retrieve the data. We only need to evaluate the test set.

    _, dataset_test = select_dataset(args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, args.batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    model = SlotModel(args)
    checkpoint = torch.load(f"{args.output_dir}/" + args.model_name, map_location=args.device)

    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # Calculate the metrics.
    all_preds = None
    all_labels = None
    all_pred_probs = None
    for batch in data_loader_test:
        imgs = batch[0].to(device, dtype=torch.float32)
        labels = batch[1].to(device, dtype=torch.int8)

        if all_labels is None:
            all_labels = labels
        else:
            all_labels = torch.cat((all_labels, labels), 0)

        preds, pred_probs = obtain_preds(model, imgs)

        if all_preds is None:
            all_preds = preds
            all_pred_probs = pred_probs
        else:
            all_preds = torch.cat((all_preds, preds), 0)
            all_pred_probs = torch.cat((all_pred_probs, pred_probs), 0)

    metrics = calc_metrics(all_labels, all_preds, all_pred_probs, args.negative, args.model_name)
    print(metrics)
