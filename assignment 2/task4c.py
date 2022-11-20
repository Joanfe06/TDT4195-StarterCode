import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real**2 + imag**2)


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)

    fft_im = np.fft.fft2(im)

    fft_kernel = np.ones(im.shape)

    """fft_im_vis = np.fft.fftshift(fft_im)
    fft_im_vis = np.log(magnitude(fft_im_vis) + 1)
    for i, x in enumerate(fft_im_vis[fft_kernel.shape[0]//2]):
        if fft_im_vis[fft_kernel.shape[0]//2][i] > 3.0:
            print(str(i) + ":" + str(x))"""

    for i in [0, 29, 87, 116, 174, 203, 261, 290, 348, 377, 435]:
        rr, cc = skimage.draw.disk((fft_kernel.shape[0]//2, i), 3)
        fft_kernel[rr, cc] = 0
    fft_kernel = np.fft.fftshift(fft_kernel)

    fft_filtered_im = fft_im * fft_kernel

    im_filtered = np.fft.ifft2(fft_filtered_im).real

    # Use plt.subplot to place two or more images beside eachother
    plt.figure(figsize=(20, 4))
    # plt.subplot(num_rows, num_cols, position (1-indexed))
    plt.subplot(1, 5, 1)
    plt.imshow(im, cmap="gray")
    plt.subplot(1, 5, 2)
    # Visualize FFT
    fft_im_vis = np.fft.fftshift(fft_im)
    fft_im_vis = np.log(magnitude(fft_im_vis) + 1)
    plt.imshow(fft_im_vis, cmap="gray")
    plt.subplot(1, 5, 3)
    # Visualize FFT kernel
    fft_kernel_vis = np.fft.fftshift(fft_kernel)
    fft_kernel_vis = np.log(magnitude(fft_kernel_vis) + 1)
    plt.imshow(fft_kernel_vis, cmap="gray")
    plt.subplot(1, 5, 4)
    # Visualize filtered FFT image
    fft_filtered_im_vis = np.fft.fftshift(fft_filtered_im)
    fft_filtered_im_vis = np.log(magnitude(fft_filtered_im_vis) + 1)
    plt.imshow(fft_filtered_im_vis, cmap="gray")
    plt.subplot(1, 5, 5)
    # Visualize filtered spatial image
    plt.imshow(im_filtered, cmap="gray")
    plt.savefig(utils.image_output_dir.joinpath("task4c.png"))
    """plt.figure(figsize=(20,20))
    plt.plot(fft_im_vis[fft_im.shape[0]//2])"""
    plt.show()
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
