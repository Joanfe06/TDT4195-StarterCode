import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    return np.dot(im[..., :3], [0.212, 0.7152, 0.0722])


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    return np.subtract(255.0, im)


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")
im_inverted = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_inverted.jpg"), im_inverted, cmap="gray")
plt.imshow(im_inverted, cmap="gray")
