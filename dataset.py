from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
from torchvision import transforms


def load_dataset(batch_size):
    """ Load and return the train, val and test dataloaders of the PCAM dataset. """
    # Other transforms could be added here.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets.
    train_set = PCAM('data', 'train', transform=transform, download=True)
    val_set = PCAM('data', 'val', transform=transform, download=True)
    test_set = PCAM('data', 'test', transform=transform, download=True)

    # Create dataloaders. Only shuffle train set.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
