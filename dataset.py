from torch.utils.data import Dataset 
import os
import pickle
import cv2 
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
    def __init__(self, root, train=True):
        self.root=root 
        if train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1,6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        
        self.images = []
        self.labels = []

        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

        # self.transform = transform

        print(len(self.images))
        print(len(self.labels))

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, item):
        image = self.images[item].reshape((3,32,32)).astype(np.float32)
        label = self.labels[item]
        return image/255, label 

if __name__  ==  "__main__":
    dataset = MyDataset(root="./cifar/cifar-10-batches-py", train=True)
    image, label = dataset.__getitem__(1000)
    image = np.reshape(image, (32,32,3))

    print(image.shape)
    print(label)
    cv2.imshow("image", cv2.resize(image, (320,320)))
    cv2.waitKey(0)