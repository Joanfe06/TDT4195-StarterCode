import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import utils


def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real**2 + imag**2)


def convolve_im(im: np.array,
                fft_kernel: np.array,
                idx,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution
    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    fft_im = np.fft.fft2(im)
    fft_filtered = fft_im * fft_kernel
    conv_result = np.fft.ifft2(fft_filtered).real

    if verbose:
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
        fft_filtered_vis = np.fft.fftshift(fft_filtered)
        fft_filtered_vis = np.log(magnitude(fft_filtered_vis) + 1)
        plt.imshow(fft_filtered_vis, cmap="gray")
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")
        plt.savefig(utils.image_output_dir.joinpath("plot" + str(idx) + ".png"))
    # END YOUR CODE HERE #
    return conv_result


if __name__ == "__main__":
    verbose = True
    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(
        im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass, 0,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(
        im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass, 1,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
