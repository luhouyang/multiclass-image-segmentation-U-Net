import os

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class CityscapesDataset(Dataset):

    def __init__(self,
                 split,
                 root_dir,
                 target_type='semantic',
                 mode='fine',
                 relabelled=True,
                 transform=None,
                 eval=False):
        super().__init__()

        self.transform = transform

        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'

        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        # prepare list of all labelTrainIds and ground truth images.
        # setting relabbelled=True recommended

        self.label_path = os.path.join(
            os.getcwd(), root_dir + '/' + self.mode + '/' + self.split)
        self.rgb_pth = os.path.join(os.getcwd(),
                                    root_dir + '/leftImg8bit/' + self.split)

        city_list = os.listdir(self.label_path)
        for city in city_list:
            temp = os.listdir(self.label_path + '/' + city)
            list_items = temp.copy()

            # 19-class label items (remove files that don't end in 'labelTrainIds.png')
            for item in temp:
                if not item.endswith('labelTrainIds.png', 0, len(item)):
                    list_items.remove(item)

            # concat full path to label images
            list_items = ['/' + city + '/' + path for path in list_items]

            self.yLabel_list.extend(list_items)
            self.XImg_list.extend([
                '/' + city + '/' + path
                for path in os.listdir(self.rgb_pth + '/' + city)
            ])

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        image = Image.open(self.rgb_pth + self.XImg_list[index])
        y = Image.open(self.label_path + self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        y = np.array(y)
        y = torch.from_numpy(y)

        y = y.type(torch.LongTensor)
        if self.eval:
            return image, y, self.XImg_list[index]
        else:
            return image, y
