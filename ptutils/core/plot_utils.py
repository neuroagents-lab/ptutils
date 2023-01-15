import matplotlib.pyplot as plt
import numpy as np


def plot_curves_from_ckpt(cpt, title=None, key="accs_top1", fontsize=15):
    num_epochs = len(cpt["results"][key]["train"])
    assert num_epochs == cpt["epoch"] + 1
    plt.plot(
        np.arange(0, num_epochs),
        cpt["results"][key]["train"],
        color="b",
        label="train set",
    )
    plt.plot(
        np.linspace(0, num_epochs, len(cpt["results"][key]["val"])),
        cpt["results"][key]["val"],
        color="r",
        label="test set",
    )
    plt.legend(loc="best")
    plt.ylabel(f"{key}", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.show()
