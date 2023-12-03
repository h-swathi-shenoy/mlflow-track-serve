from torchvision import transforms


class ImageTransform:
    def __init__(self):
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.data_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(0.2),
             transforms.ToTensor(),
             transforms.Normalize(self.mean, self.std)]
        )

    def __call__(self, img):
        return self.data_transform(img)
