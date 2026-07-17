from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EUROSAT_CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EuroSATTorchDataset(Dataset):
    """Wraps a HuggingFace EuroSAT split as a PyTorch Dataset with image transforms applied."""

    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item["image"].convert("RGB")), item["label"]


def build_image_transforms(image_size: int = 224) -> transforms.Compose:
    """Build a resize + ToTensor + Normalize pipeline for the given image_size; returns Compose."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_eurosat_splits(smoke_test: bool = False):
    """Load blanchon/EuroSAT_RGB from HuggingFace Hub; returns (train, validation, test) HF splits."""
    ds = load_dataset("blanchon/EuroSAT_RGB")
    train_split = ds["train"]
    val_split = ds["validation"]
    test_split = ds["test"]
    if smoke_test:
        train_split = train_split.select(range(10))
        val_split = val_split.select(range(10))
        test_split = test_split.select(range(10))
    return train_split, val_split, test_split


def make_dataloaders(train_split, val_split, test_split, batch_size: int, image_size: int = 224):
    """Wrap HuggingFace splits in PyTorch DataLoaders; returns (train_loader, val_loader, test_loader)."""
    transform = build_image_transforms(image_size)
    train_loader = DataLoader(
        EuroSATTorchDataset(train_split, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        EuroSATTorchDataset(val_split, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        EuroSATTorchDataset(test_split, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader
