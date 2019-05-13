import cv2
import numpy as np
from matplotlib import pyplot as plt


def FFT_Exercice(image_path):
    img = cv2.imread(image_path, 0)

    # FFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Low Pass
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur, cmap='gray')
    plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])

    plt.show()

    # High PASS
    filtered = img - blur
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered, cmap='gray')
    plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
    plt.show()
