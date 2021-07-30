from pathlib import Path
from typing import Union, Optional, Callable, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FlowerDataSet(Dataset):
    """
    Custom DataSet for the Flower data
    """
    def __init__(self, data_dir: Union[str, Path], transformation: Optional[Callable] = None) -> None:
        super().__init__()
        if type(data_dir) is str:
            data_dir = Path(data_dir)

        if transformation is None:
            transformation = transforms.ToTensor()

        self.transformation = transformation

        self.files = list(data_dir.rglob("*.jpg"))

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """
        Overwrite the getitem method:
        * load image as Pillow image and transform to Tensor
        * get image label from file name

        Parameters
        ----------
        item: int

        Returns
        -------
        Tuple[Tensor, Tensor]
            Tuple of [image, label]
        """
        # Load image
        fn = self.files[item]
        image = Image.open(fn).convert("RGB")
        tensor_image = self.transformation(image)

        # Derive class from filename. For now I use the -1 as a hack as the label start at 1. However,
        # this needs to be implemented carefully when adding the text labels!
        label = int(fn.parent.name) - 1
        return tensor_image, torch.tensor(label)

    def __len__(self):
        return len(self.files)


def get_loader(base_dir: Union[str, Path], phase: str, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    """
    Get the data loader for the chosen phase (train/test/valid)

    Parameters
    ----------
    base_dir: Union[str, Path]
    phase: str
    batch_size: int, default = 1
    num_workers: int, default = 0

    Returns
    -------
    DataLoader
    """
    phases = ['train', 'test', 'valid']
    assert phase in phases, f"Parameter phase needs to one of {phases}"

    if type(base_dir) is str:
        base_dir = Path(base_dir)

    dataset = FlowerDataSet(base_dir / phase, transformation=DATA_TRANSFORMS[phase])
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers)
    return loader


if __name__ == '__main__':
    train_dir = 'data/flowers/train'
    print(train_dir)
    flowers = FlowerDataSet(train_dir, transformation=DATA_TRANSFORMS['train'])

    img, lbl = flowers[0]
    print(img.shape)
    print(lbl)

    flower_loader = DataLoader(flowers, batch_size=4, shuffle=True)

    images, labels = next(iter(flower_loader))
    print(images.shape)
    print(labels)
