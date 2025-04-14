import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTWrapper(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, transform=None):
        """
        MNIST Dataset Wrapper
        
        Args:
            root (str): Root directory to store the MNIST data.
            train (bool): If True, load training data; otherwise load test data.
            transform (callable, optional): Transformation to apply to the data.
        """
        self.transform = transform
        self.dataset = datasets.MNIST(root=root, train=train, download=True)
        
        self.num_classes = 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data to retrieve.

        Returns:
            Tuple[Tensor, int]: Transformed image and its label.
        """
        img, label = self.dataset[index]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.dataset)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTWrapper(root="./data", train=True, transform=transform)

    test_dataset = MNISTWrapper(root="./data", train=False, transform=transform)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for images, labels in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        break
