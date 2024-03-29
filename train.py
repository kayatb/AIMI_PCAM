import dataset

import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from utils import optional_tqdm
import torchvision

class TTA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor):
        pred = sum([
            self.base(x.rot90(rotation, dims=(-1, -2)))
            for rotation in range(4)
        ]) / 4

        return pred

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

def directional_weight_decay(parameters, base_parameters, strength):
    for p, base_p in zip(parameters, base_parameters):
        # base_p: nn.parameter.Parameter = base_p
        if p.requires_grad:
            p.data.add_(base_p - p, alpha=strength)

def train(model: nn.Module, args, startup_params=None, weight_decay_base=None, weight_decay_strength=0.001):
    """
    Performs a full training cycle of a PCAM model.

    Returns:
      model: The trained model that performed best on the validation set.
      logging_info: An object containing logging information: train loss, train accuracy,
                     validation accuracy, final test accuracy, number of epochs (feel free to add anything you want).
    """

    # Set the random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    dataloader_train, dataloader_val, dataloader_test = dataset.load_datasets(args.batch_size)
    print("Data loaded.")

    # Initialize model and loss module
    model = model.to(device)
    if isinstance(weight_decay_base, nn.Module):
        weight_decay_base = weight_decay_base.to(device)
    loss_module = torch.nn.BCEWithLogitsLoss()

    logging_info = {}
    logging_info['train_loss'] = []
    logging_info['train_acc'] = []
    logging_info['val_acc'] = []
    logging_info['no_epochs'] = args.epochs

    best_val_acc = 0
    # best_model = None

    # Training loop including validation
    if startup_params is not None:
        optimizer = optim.Adam(startup_params(model), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.file_name is None:
        save_filename = os.path.join(args.output_dir, f"{args.model}_best_model")
    else:
        save_filename = args.file_name

    if args.load_checkpoint:
        model.load_state_dict(torch.load(save_filename + ".pt"))

    print("=== Starting the training ===")
    for e in range(args.epochs):
        model.train()  # Put model in train mode

        loss_value = 0
        batches_count = 0

        train_correct = 0
        train_incorrect = 0

        data_iterator = optional_tqdm(dataloader_train, args)
        for x, labels in data_iterator:
            batches_count += 1

            if args.max_batches_per_epoch is not None and batches_count >= args.max_batches_per_epoch:
                break

            binary_labels = labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=2).type(torch.FloatTensor)

            # Necessary when running on GPU.
            x = x.to(device)
            labels = labels.to(device)

            predictions = model(x).squeeze(dim=1)  # Forward pass
            loss = loss_module(predictions, labels)  # Calculate loss

            optimizer.zero_grad()
            loss.backward()
            loss_value += loss.item()

            predicted_labels = torch.argmax(predictions, dim=-1)
            train_correct += torch.sum(predicted_labels == binary_labels).item()
            train_incorrect += torch.sum(predicted_labels != binary_labels).item()

            # Update parameters
            optimizer.step()

            if args.tqdm:
                train_acc = train_correct / (train_correct + train_incorrect) # evaluate_model(model, dataloader_train, device)
                data_iterator.set_description(f"train_acc = {train_acc:.3f}")

            if weight_decay_base == 0:
                optimizer.zero_grad()
                for parameter in model.parameters():
                    parameter.data.add_(-parameter, alpha=weight_decay_strength)
            elif isinstance(weight_decay_base, torchvision.models.ResNet) and isinstance(model, torchvision.models.ResNet):
                optimizer.zero_grad()
                # weight_decay_base: torchvision.models.ResNet = weight_decay_base
                # directional_weight_decay(model.conv1.parameters(), weight_decay_base.conv1.parameters())
                directional_weight_decay(model.conv1  .parameters(), weight_decay_base.conv1  .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.bn1    .parameters(), weight_decay_base.bn1    .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.relu   .parameters(), weight_decay_base.relu   .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.maxpool.parameters(), weight_decay_base.maxpool.parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.layer1 .parameters(), weight_decay_base.layer1 .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.layer2 .parameters(), weight_decay_base.layer2 .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.layer3 .parameters(), weight_decay_base.layer3 .parameters(), strength=weight_decay_strength)
                directional_weight_decay(model.layer4 .parameters(), weight_decay_base.layer4 .parameters(), strength=weight_decay_strength)

                # for parameter, base_value in zip(model.parameters(), weight_decay_base.parameters()):
                #     parameter.add_(-parameter, base_value, alpha=weight_decay_strength)
                
        loss_value /= batches_count  # Average over all batches.
        logging_info['train_loss'].append(loss_value)
        # Calculate validation accuracy for this epoch
        val_acc = evaluate_model(model, dataloader_val, device)
        logging_info['val_acc'].append(val_acc)
        # Save train set accuracy.
        train_acc = train_correct / (train_correct + train_incorrect) # evaluate_model(model, dataloader_train, device)
        logging_info['train_acc'].append(train_acc)

        print(f"Epoch {e + 1} => training accuacy: {train_acc}, validation accuracy: {val_acc}, loss: {loss_value}")

        # Update best model (if necessary)
        if val_acc > best_val_acc:
            # best_model = deepcopy(model)
            torch.save(model.state_dict(), save_filename + ".pt")
            print(f"Saved the best performing model at: {save_filename}.pt")

            best_val_acc = val_acc

        if startup_params is not None:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Test best model
    model.load_state_dict(torch.load(save_filename + ".pt"))
    print("=== Finished training, now evaluating on the test set ===")
    test_accuracy = evaluate_model(model, dataloader_test, device)
    logging_info['test_acc'] = test_accuracy
    print(f"Final Test Set Accuracy: {test_accuracy}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        torch.save(model.state_dict(), save_filename + ".pt")
        print(f"Saved the best performing model at: {save_filename}.pt")

        with open(save_filename + ".json", 'w') as fp:
            json.dump(logging_info, fp)
        print(f"Saved logging info at: {save_filename}.json")

    return model, logging_info
