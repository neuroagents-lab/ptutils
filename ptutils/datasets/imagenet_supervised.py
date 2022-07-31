from ptutils.datasets.imagenet_base import ImageNetBase

__all__ = ["ImageNetSupervised"]


class ImageNetSupervised(ImageNetBase):
    """
    ImageNet data set class that returns an image and its label.

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        super(ImageNetSupervised, self).__init__(
            is_train=is_train,
            imagenet_dir=imagenet_dir,
            image_transforms=image_transforms,
        )

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


if __name__ == "__main__":
    from torchvision import transforms
    from ptutils.core.default_dirs import IMAGENET_DATA_DIR

    my_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    d = ImageNetSupervised(
        is_train=True, imagenet_dir=IMAGENET_DATA_DIR, image_transforms=my_transforms
    )
    image, label = d[1000000]
    print(image.shape)
    print(label)
