import pytorch_lightning as plt
import os
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as pl
from image_transform import ImageTransform
from PIL import Image

current_dir = Path(__file__).absolute().parent.parent
data_dir = current_dir.joinpath('data/cat-dogs')


class CatDogDataLoader(plt.LightningDataModule):
    def __init__(self, image_path: Path):
        self.img_path = image_path
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.transform = ImageTransform()
        class_dir = os.listdir(self.img_path)
        self.data = []
        for class_ in class_dir:
            class_img = os.listdir(self.img_path.joinpath(class_))
            label = class_
            for img in class_img:
                self.data.append([self.img_path.joinpath(class_).joinpath(img), label])
        self.class_map = {'cat':0, 'dog':1}
        self.img_dim = (224,224)

    def __len__(self):
        '''
        Returns length of the full dataset loaded
        :return:
        '''
        return len(self.data)

    def __getitem__(self, item):
        '''
        Retreives indiviual images and its labels for further tuning of Images
        :param item: individual images
        :return:
        img_transform: transformed image
        label : class labels
        '''
        img, label = self.data[item]
        img_ = Image.open(img)
        img_transform = self.transform(img_)
        label = self.class_map[label]
        return img_transform, label


if __name__ == "__main__":
    full_data = CatDogDataLoader(data_dir)
    data_loader = DataLoader(full_data, batch_size=1, shuffle=True)
    for count in range(0,3):
        img = next(iter(data_loader))
        img_sq = img[0].squeeze()
        pl.imshow(img_sq.permute(1, 2, 0))
        pl.show()












