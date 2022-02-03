import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import struct
from PIL import Image
import numpy as np
from conf import n_epochs, batch_size_train, batch_size_test, learning_rate, momentum, log_interval, random_seed
from conf import train_images_idx3_ubyte_file, train_labels_idx1_ubyte_file, test_images_idx3_ubyte_file, test_labels_idx1_ubyte_file

"""
There are two ways of loading the MNIST data:
    1. Decode the binary file manually and customize the data loader
    2. Using function Load_TrainData and Load_TestData, which are defined using embedded function of Pytorch

Data structure of the MNIST dataset is:
images
[offset] [type]          [value]          [description]                     labels
0000     32 bit integer  0x00000803(2051) magic number                      0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of images                  0004     32 bit integer  60000            number of items
0008     32 bit integer  28               number of rows                    0008     unsigned byte   ??               label
0012     32 bit integer  28               number of columns                 0009     unsigned byte   ??               label
0016     unsigned byte   ??               pixel                             ........
0017     unsigned byte   ??               pixel                             xxxx     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               pixel

pixel value ranges from 0 to 255. label value ranges from 0 to 9.
one pixel is a byte. one label is also a byte.
"""

############################ First Way of Loading ############################
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()       # read bin file

    # obtain magic number, number of images, width of images and height of images
    offset = 0                      # read from 0 byte position
    fmt_header = '>iiii'            # read four int data in MSB mode (1 int = 4 byte)
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d\nnumber of images:%d\nsize of images:%d×%d\n'% (magic_number, num_images, num_rows, num_cols))

    # obtain images
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)       # now offset should be 16 bytes
    fmt_image = '>' + str(image_size) + 'B'     # One image requires reading 28*28=784 pixels or bytes
    images = np.empty((num_images, num_rows, num_cols))

    # convert to numpy array
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('Unpack %dth images' % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()       # read bin file

    # obtain magic number and number of labels
    offset = 0                      # read from 0 byte position
    fmt_header = '>ii'              # read 2 int data in MSB mode (1 int = 4 byte)
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic number:%d\nnumber of labels: %d' % (magic_number, num_images))

    # obtain labels
    offset += struct.calcsize(fmt_header)       # now offset should be 8 bytes
    fmt_image = '>B'                            # One label requires reading one byte
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('Unpack %d labels' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = torch.from_numpy(images).to(torch.uint8)
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img   = Image.fromarray(self.images[index].numpy(), mode='L')
        label = int(self.labels[index])
        img   = self.transform(img)       # some bug occurs if a ndarray is transformed (don't know why)
        return img, label
    def __len__(self):
        return len(self.images)




############################ Second Way of Loading ############################

def Load_TrainData():
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=True, download=True,
             transform=torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(
                 (0.1307,), (0.3081,))
             ])), batch_size=batch_size_train, shuffle=True)


def Load_TestData():
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=False, download=True,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
             (0.1307,), (0.3081,))
            ])), batch_size=batch_size_test, shuffle=True)



