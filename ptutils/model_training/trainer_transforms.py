"""
Contains information on image transformations used by different trainers
(e.g. supervised, self-supervised, etc) during training and validation.
"""
import numpy as np
from torchvision import transforms
from ptutils.core.default_constants import IMAGENET_MEAN, IMAGENET_STD

def compose_ifnot(dataloader_transforms):
    if (dataloader_transforms is not None) and isinstance(dataloader_transforms, list):
        return transforms.Compose(dataloader_transforms)
    else:
        assert type(dataloader_transforms).__name__ == 'Compose'
        return dataloader_transforms

TRAINER_TRANSFORMS = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
