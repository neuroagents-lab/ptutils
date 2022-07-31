"""
Contains information on image mean and standard deviation for RGB channels
that models were trained on.

The format is as follows: each model name is associated with a dictionary
that has "mean", "std", "input_size" as keys.  "input_size" is the image size
the model was trained on.
"""
from ptutils.model_training.trainer_transforms import TRAINER_TRANSFORMS

MODEL_TRANSFORMS = {
    "resnet18": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet34": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet50": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet101": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet152": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
}
