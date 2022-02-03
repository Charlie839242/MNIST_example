from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def mean_std(path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])         # Do Nothing

    # Load Data
    train_dataset = datasets.MNIST(root=path, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Mean and std
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        x = inputs.view(-1, 28 * 28)
        x_std = x.std().item()
        x_mean = x.mean().item()

        print('mean: ' + str(round(x_mean, 4)))
        print('std: ' + str(round(x_std, 4)))


mean, std = mean_std('./dataset', 60000)
