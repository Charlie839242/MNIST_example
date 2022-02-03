import torchvision
from models import Net
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import os
from conf import n_epochs, batch_size_train, batch_size_test, learning_rate, momentum, log_interval, random_seed
from conf import train_images_idx3_ubyte_file, train_labels_idx1_ubyte_file, test_images_idx3_ubyte_file, test_labels_idx1_ubyte_file
from dataset import Load_TrainData, Load_TestData, CustomDataset, decode_idx3_ubyte, decode_idx1_ubyte


torch.manual_seed(random_seed)      # Choose a seed for the process


# 1. Downloading Dataset
if(os.path.exists('./dataset/MNIST')):
    print('MNIST Dataset Already Downloaded')
else:
    print('Downloading MNIST dataset ......')
    mnist = torchvision.datasets.MNIST(root='./dataset', train=True, download=True)


# 2. Preprocess
"""First way of loading dataset using built-in function of PyTorch"""
# train_loader = Load_TrainData()
# test_loader  = Load_TestData()

"""Second way of loading dataset using customized class 'CustomDataset' """
train_loader = torch.utils.data.DataLoader(
                CustomDataset(images = decode_idx3_ubyte(train_images_idx3_ubyte_file),
                              labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file),
                              transform = torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))
                              ])), batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                CustomDataset(images = decode_idx3_ubyte(test_images_idx3_ubyte_file),
                              labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file),
                              transform = torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))
                              ])), batch_size=batch_size_test, shuffle=True)


# 3. Train and Test
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):   # 640 per batch
        # data type: torch.float32      target type: torch.int64
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './models/model.pth')
            torch.save(optimizer.state_dict(), './models//optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.type(torch.float32)
            # target = target.type(torch.int64)
            # print('data shape:', data.shape, '\ntarget shape:', target.shape)
            # print(data.dtype, target.dtype)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# store losses in txt files
f_train = open('./img/train_loss.txt', 'w')
for i in range(len(train_losses)):
    f_train.write(str(train_counter[i]) + ' ' + str(train_losses[i]) + '\n')
f_train.close()

f_train = open('./img/test_loss.txt', 'w')
for i in range(len(test_losses)):
    f_train.write(str(test_counter[i]) + ' ' + str(test_losses[i]) + '\n')
f_train.close()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig('./img/loss.jpg')
plt.show()



