# 我们读取图片的根目录， 在根目录下有所有图片的txt文件， 拿到txt文件后， 先读取txt文件， 之后遍历txt文件中的每一行， 首先去除掉尾部的换行符， 在以空格切分，前半部分是图片名称， 后半部分是图片标签， 当图片名称和根目录结合，就得到了我们的图片路径
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transforms = transforms.Compose([
    transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(256),  # 将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])


class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path

        self.txt_root = self.root + '//' + 'data.txt'

        f = open(self.txt_root, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            # print(word[0], word[1], word[2])
            # word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
            imgs.append(os.path.join(self.root, word[2], word[0]))

            labels.append(word[1])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')

        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label


if __name__ == '__main__':
    path = r'../Data'
    dataset = MyDataset(path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):
        imgs, labels = data
        print(imgs.shape, labels.shape)
        print(labels)
        # print(imgs)
        # print(labels)
        # print(imgs[0].shape)
        # print(imgs[0])
        plt.imshow(imgs[0].permute(1, 2, 0))
        plt.show()
        break
