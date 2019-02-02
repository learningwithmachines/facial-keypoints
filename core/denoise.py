import cv2
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt


def fftdenoise_ch(noisy_channel: np.ndarray,
                  keep_fraction: float=0.1,
                  passes: int = 1)-> np.ndarray:
    for i in range(0, passes):
        gpuarray = cp.asarray(noisy_channel)
        im_fft = cp.fft.fft2(gpuarray)
        im_fft2 = im_fft.copy()
        r, c = im_fft2.shape
        im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
        im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
        im_new = cp.fft.ifft2(im_fft2).real

    im_new = cp.asnumpy(im_new).astype(np.uint8)
    return im_new



def rgb_denoise(noisy_rgb:np.ndarray,
                keep_frac: float=0.1,
                passes: int=1,
                flipch:bool=False)->np.ndarray:

    rgb = cv2.merge([fftdenoise_ch(ch_,keep_frac,passes) for ch_ in cv2.split(noisy_rgb)])
    if flipch:
        return rgb[:, :, ::-1]
    else:
        return rgb



if __name__ == '__main__':

    # from: https://www.scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

    testimg = np.asarray(cv2.imread('..\\images\\gus.jpg', 1))

    # make some noise

    noisy_image = np.random.randn(testimg.shape[0], testimg.shape[1], testimg.shape[2]) * 32
    noisy_image = testimg.astype(np.uint8) + noisy_image.astype(np.uint8)

    plt.figure()
    plt.imshow(noisy_image[:, :, ::-1], cmap='gray')
    plt.title('Noisy Image')

    plt.figure()
    plt.imshow(rgb_denoise(noisy_image), cmap='gray')
    plt.title('Reconstructed Image')
