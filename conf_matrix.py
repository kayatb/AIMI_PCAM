import argparse
import torch
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score

from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
from torchvision import transforms


def load_datasets(batch_size):
    """Load and return the train, val and test dataloaders of the PCAM dataset."""
    # Other transforms could be added here.
    transform = transforms.Compose([transforms.ToTensor()])

    # Create datasets.
    # WARNING: set `download=True` if you don't have the dataset yet. Won't download if it is present.
    train_set = PCAM("data", "train", transform=transform, download=True)
    val_set = PCAM("data", "val", transform=transform, download=True)
    test_set = PCAM("data", "test", transform=transform, download=True)

    # Create dataloaders. Only shuffle train set.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader


def add_args():
    """ Add a lot of commande line arguments for specifying all sorts of hyperparams and other settings. """
    parser = argparse.ArgumentParser("Set PCAM model", add_help=False)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--output_dir", default="saved_models/", help="path where to save trained model, empty for no saving")
    return parser

def calc_metrics(preds, labels, pred_probs):
    # Calculate the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Calculate the metrics.
    metrics = {}
    metrics["auc"] = roc_auc_score(labels, pred_probs)
    metrics["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    metrics["recall"] = tp / (tp + fn)
    metrics["precision"] = tp / (tp + fp)
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    metrics["kappa"] = cohen_kappa_score(labels, preds)

    return metrics

def evaluate_model(model, data_loader, device):
    """
    Performs the evaluation of the PCAM model on a given dataset.

    Args:
      model: An instance of the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.
    """
    preds_whole_set = []
    preds_argmax_whole_set = []
    labels_whole_set = []

    model.eval()  # Put model in evaluation mode.

    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            labels = labels.to(device)

            predictions = model(x).squeeze(dim=1)
            preds = np.argmax(predictions.cpu().numpy(), axis=1)
            for pred_argmax, label, pred in zip(preds, labels.cpu().numpy(), predictions[:, 1].cpu().numpy()):
                preds_whole_set.append(pred)
                preds_argmax_whole_set.append(pred_argmax)
                labels_whole_set.append(label)
    return preds_argmax_whole_set, labels_whole_set, preds_whole_set

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Calculate metrics for model", parents=[add_args()])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_train, dataloader_val, dataloader_test = load_datasets(args.batch_size)
    model_name = os.path.join(args.output_dir, f"{args.model}_best_model.pt")
    model = torch.load(model_name)
    preds, labels, pred_probs = evaluate_model(model, dataloader_test, device)
    print(calc_metrics(preds, labels, pred_probs))