from imageio import imread
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


GRAY_SHAPE = 2
RGB_SHAPE = 3
GRAY_REP = 1
RGB_REP = 2
NORM_FACTOR = 255
HIST_SIZE = 256
YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
FILE_DOESNT_EXIST = "The file doesn't exist."
FALSE_REP = "The representation is not legal, it should be 1 or 2."
ILLEGAL = "Illegal input."


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation.
    :param filename: The image filename.
    :param representation: Representation code - (1) gray scale image (2) RGB image.
    :return: The converted image normalized to the range [0, 1].
    """
    try:
        im = imread(filename)
    except FileNotFoundError:
        print(FILE_DOESNT_EXIST)
        return
    if representation != GRAY_REP and representation != RGB_REP:
        print(FALSE_REP)
        return

    if len(im.shape) == RGB_SHAPE and representation == GRAY_REP:
        im = rgb2gray(im)
        return im.astype(np.float64)
    im_float = im.astype(np.float64)
    im_float /= NORM_FACTOR
    return im_float


def imdisplay(filename, representation):
    """
    Display an image in a given representation.
    :param filename: The image filename.
    :param representation: Representation code - (1) gray scale image (2) RGB image.
    """
    im = read_image(filename, representation)
    if not im:
        return

    if representation == GRAY_REP:
        plt.figure()
        plt.imshow((im * NORM_FACTOR).astype(np.uint8), cmap=plt.cm.gray)
    elif representation == RGB_REP:
        plt.figure()
        plt.imshow((im * NORM_FACTOR).astype(np.uint8))
    else:
        print(FALSE_REP)
        return
    plt.show()


def multiply_im_mat(im, mat):
    """
    Multiply each RGB/YIQ vector in the given image with the given matrix.
    :param im: RGB or YIQ image.
    :param mat: Matrix to multiply with.
    :return: The image matrix after the multiplication.
    """
    first, second, third = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    first_new = mat[0][0] * first + mat[0][1] * second + mat[0][2] * third
    second_new = mat[1][0] * first + mat[1][1] * second + mat[1][2] * third
    third_new = mat[2][0] * first + mat[2][1] * second + mat[2][2] * third
    yiq_mat = np.stack((first_new, second_new, third_new), 2)
    return yiq_mat.astype(np.float64)


def rgb2yiq(imRGB):
    """
    Transform the given RGB image into the YIQ color space.
    :param imRGB: The RGB image to transform.
    :return: The transformed YIQ image.
    """
    return multiply_im_mat(imRGB, YIQ_MAT)


def yiq2rgb(imYIQ):
    """
    Transform the given YIQ image into the RGB color space.
    :param imYIQ: The YIQ image to transform.
    :return: The transformed RGB image.
    """
    inv_mat = np.linalg.inv(YIQ_MAT)
    return multiply_im_mat(imYIQ, inv_mat)


def get_grayscale_yiq_hist(im):
    """
    Calculates the gray scale of the given image, the YIQ transformed image (None is the given image
    is in gray color, the histogram of the gray scale and the cumulative histogram.
    :param im: The image matrix (RGB or gray scale.
    :return: Gray scale, YIQ image, histogram, cumulative histogram.
    """
    if len(im.shape) == RGB_SHAPE:
        yiq_im = rgb2yiq(im)
        gray_scale = yiq_im[:, :, 0]
    else:
        yiq_im = None
        gray_scale = im
    gray_scale *= NORM_FACTOR
    hist = np.histogram(gray_scale, bins=np.arange(HIST_SIZE + 1))[0]
    cum_hist = np.cumsum(hist)
    return gray_scale, yiq_im, hist, cum_hist


def histogram_equalize(im_orig):
    """
    Calculate the histogram equalization of a given grayscale or RGB image.
    :param im_orig: image to equalize its histogram.
    :return: The equalized image matrix, the original image histogram, the equalized image
             histogram.
    """
    gray_scale, yiq_im, hist_orig, cum_hist = get_grayscale_yiq_hist(im_orig)
    no_zero = np.nonzero(cum_hist)[0][0]
    cum_hist = (cum_hist - cum_hist[no_zero]) / (cum_hist[-1] - cum_hist[no_zero])
    cum_hist *= NORM_FACTOR
    cum_hist = np.round(cum_hist)
    new_gray_scale = cum_hist[gray_scale.astype(np.int64)] / NORM_FACTOR
    hist_eq = np.histogram(new_gray_scale, bins=np.arange(HIST_SIZE + 1))[0]
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im[:, :, 0] = new_gray_scale
        return yiq2rgb(yiq_im), hist_orig, hist_eq
    return new_gray_scale, hist_orig, hist_eq


def init_z(cum_hist, n_quant):
    """
    Calculates the initialized z array for the quantization.
    :param cum_hist: The cumulative histogram.
    :param n_quant: The number of intensities.
    :return: The initialized z array.
    """
    pixels_in_z = np.round(cum_hist[-1] / n_quant)
    z_arr = np.zeros(n_quant + 1).astype(int)
    for i in range(1, n_quant):
        z_arr[i] = np.searchsorted(cum_hist, pixels_in_z * i)
    z_arr[-1] = NORM_FACTOR
    return z_arr


def calc_q_z_error(z_arr, hist, n_quant, n_iter):
    """
    Calculates the final z array, q array and error array.
    :param z_arr: The initialized z array.
    :param hist: The original image histogram.
    :param n_quant: The number of intensities.
    :param n_iter: The maximum number of iterations.
    :return: z array, q array, error array.
    """
    q_arr = np.zeros(n_quant).astype(int)
    error = []
    for i in range(n_iter):
        for j in range(n_quant):
            seg_pixels = range(z_arr[j], z_arr[j + 1] + 1)
            hist_in_seg = hist[z_arr[j]:z_arr[j + 1] + 1]
            q_arr[j] = np.round(np.average(seg_pixels, weights=hist_in_seg))
        last_z_arr = z_arr.copy()
        for j in range(1, n_quant):
            z_arr[j] = np.round((q_arr[j - 1] + q_arr[j]) / 2)
        if np.array_equal(last_z_arr, z_arr):
            break
        e = 0
        for j in range(n_quant):
            e += sum(((q_arr[j] - z) ** 2) * hist[z] for z in range(z_arr[j], z_arr[j + 1]))
        error.append(e)
    return z_arr, q_arr, error


def quantize(im_orig, n_quant, n_iter):
    """
    Calculates the optimal quantization of the given grayscale or RGB image.
    :param im_orig: The image to quantize.
    :param n_quant: The number of intensities.
    :param n_iter: The maximum number of iterations.
    :return: The quantized output image, an array of the total intensities error for each iteration
             of the quantization procedure.
    """
    if n_quant <= 0 or n_iter < 0:
        print(ILLEGAL)
        return
    if n_iter == 0:
        return im_orig, []

    gray_scale, yiq_im, hist, cum_hist = get_grayscale_yiq_hist(im_orig.copy())
    init_z_arr = init_z(cum_hist, n_quant)
    z_arr, q_arr, error = calc_q_z_error(init_z_arr, hist, n_quant, n_iter)
    new_hist = np.zeros(HIST_SIZE).astype(np.int64)
    for i in range(n_quant):
        np.put(new_hist, range(z_arr[i], z_arr[i+1] + 1), q_arr[i])
    new_gray_scale = new_hist[gray_scale.astype(np.int64)] / NORM_FACTOR
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im[:, :, 0] = new_gray_scale
        return yiq2rgb(yiq_im), error
    return new_gray_scale, error
