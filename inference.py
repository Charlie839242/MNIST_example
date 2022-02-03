from models import Net
import torchvision
import torch
from dataset import Load_TestData
from PIL import Image


def convertjpg(jpgfile, width=28, height=28):
    img = Image.open(jpgfile)
    img = img.convert('L')
    new_img = img.resize((width,height),Image.BILINEAR)
    return new_img

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))])

test_loader = Load_TestData()

# Load the pth Model
network = Net()
network_state_dict = torch.load('./models/model.pth')
network.load_state_dict(network_state_dict)

# resize the test img
img = convertjpg('./img/test.jpg')
img = transform(img).reshape(1, 1, 28, 28)
output = network(img)
_, pred = torch.max(output, 1)
print(int(pred))


