from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=64, validation_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_size = int((1 - validation_split) * len(train_dataset))
    val_size   = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
