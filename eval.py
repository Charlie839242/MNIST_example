from models import Net
import torch.nn.functional as F
import torch
from dataset import Load_TestData

test_loader = Load_TestData()

# Load the pth Model
network = Net()
network_state_dict = torch.load('./models/model.pth')
network.load_state_dict(network_state_dict)
def eval():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            # print(output)
            # print(pred)
            # print(target)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

eval()