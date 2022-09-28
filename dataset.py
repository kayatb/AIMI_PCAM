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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
