"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

def itsRGB(img:np.ndarray)->bool:

    return len(img.shape)>2

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation not in [LOAD_GRAY_SCALE, LOAD_RGB]:
        print(" incorrect input of representation mode")
        exit(0)

    im = cv2.imread(filename, representation-1)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255.0

    if representation == LOAD_GRAY_SCALE:
        img= img[:,:,0]

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename,representation)

    if representation == LOAD_GRAY_SCALE :
        plt.imshow(img, cmap='gray', interpolation='bicubic')
    else:
        plt.imshow(img)

    plt.xticks([]), plt.yticks([])
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                                            [0.596, -0.275, -0.321],
                                            [0.212, -0.523, 0.311]])
    return np.dot(imgRGB, rgb_to_yiq.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    yiq_to_rgb = inv(rgb_to_yiq) # the inverse matrix

    return np.dot(imgYIQ, yiq_to_rgb.T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original image
        :return: (imgEq,histOrg,histEQ)
    """
    flag = False   # for RGB or GrayScale image

    if itsRGB(imgOrig):
        YIQ_im = transformRGB2YIQ(imgOrig)
        src = YIQ_im[:,:,0]  #   the Y channel
        flag = True
    else:
        src = imgOrig

    # normalize the picture to intensities [0,255]
    src = np.round(src*255).astype(int)

    # create histogram of original picture intensities
    histOrig, bins = np.histogram(src.flatten(), bins=256, range=[0, 255])

    # create a Cumsum of intensities
    CumSum = np.cumsum(histOrig)
    CumSumNorm_Orig_plot = CumSum * histOrig.max() /CumSum.max()

    # LookUpTable
    LUT = (CumSum * 255 / CumSum.max()).astype('uint8')

    # the equalized image
    imEq = LUT[src]

    # histogram and Cumsum of the equalized image
    histEq, bins = np.histogram(imEq.flatten(), bins=256, range=[0, 255])

    EqCumSum = np.cumsum(histEq)
    CumSumNorm_Eq_plot = EqCumSum * histEq.max() / EqCumSum.max()


    # let's plot!!!
    fig, ((histOrig_plot, histEq_plot), (Orig_im, Eq_im)) = plt.subplots(2, 2, figsize=(15,15), )

    histOrig_plot.hist(src.flatten(), bins=range(0, 255))
    histOrig_plot.plot(CumSumNorm_Orig_plot)
    histOrig_plot.set_title('histOrig and CumSum')

    histEq_plot.hist(imEq.flatten(), bins=range(0, 255))
    histEq_plot.plot(CumSumNorm_Eq_plot)
    histEq_plot.set_title('histEq and CumSum')

    if flag :
        YIQ_im[:,:,0] = imEq/255.0
        imEq = transformYIQ2RGB(YIQ_im)

        Orig_im.imshow(imgOrig)

        Eq_im.imshow(imEq)

    else:
        imEq = imEq/255.0

        Orig_im.imshow(imgOrig, cmap='gray', interpolation='bicubic')
        Eq_im.imshow(imEq, cmap='gray', interpolation='bicubic')

    Orig_im.set_title('Original image')
    Eq_im.set_title('Equalized Image')

    plt.show()

    return imEq , histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    flag = False   # for RGB or GrayScale image

    if itsRGB(imOrig):
        YIQ_im = transformRGB2YIQ(imOrig)
        src = YIQ_im[:,:,0]  #   the Y channel
        flag = True
    else:
        src = imOrig

    # normalize the picture to intensities [0,255]
    src = np.round(src*255).astype(int)

    # create histogram of original picture intensities
    histOrig, bins = np.histogram(src.flatten(), bins=256, range=[0, 255])
    CumSum = np.cumsum(histOrig)
    ListCumsum = list(CumSum)

    PixelsForEachBorder = CumSum.max()/nQuant

    img_list = list()
    error_list = list()

    # initials borders - between each border we have approximately the same number of pixels
    borders = [0]
    for i in range(1,nQuant):
        intensity = ListCumsum.index(next(filter(lambda j: j >= i*PixelsForEachBorder, CumSum)))
        borders.append(intensity)
    borders.append(255)


    colors_map = np.zeros(nQuant).astype(int)

    while nIter :
        # mapping the colors between the borders to one color
        for i in range(nQuant):
            colors_map[i] = \
                round(sum(np.arange(borders[i], borders[i+1])*histOrig[borders[i]: borders[i+1]])/
                      sum(histOrig[borders[i]: borders[i+1]]))
        # adjust the borders between 2 mapping color
        for i in range(1, nQuant) :
            borders[i] = round((colors_map[i-1]+colors_map[i])/2)

        img_quant = np.copy(src)

        # mapping each color according to the colormap
        for i in range(nQuant):
            img_quant[(img_quant>=borders[i]) & (img_quant<borders[i+1])] = colors_map[i]

        # adding to the image list
        if flag:
            YIQ_im[:, :, 0] = img_quant / 255.0
            temp = transformYIQ2RGB(YIQ_im)

            img_list.append(temp)

        else:
            img_list.append(img_quant/255)

        # compute the error
        matrixError = (src - img_quant)
        error_list.append(math.sqrt(np.sum(np.power(matrixError,2)))/CumSum.max())

        # if the error converge stop the process
        if len(error_list)>=2 :
            if error_list[-2]==error_list[-1]: break

        nIter -= 1

    plt.plot(error_list)
    plt.show()

    return img_list , error_list

if __name__ == '__main__':
    quantizeImage(imReadAndConvert("beach.jpg",2 ), 3, 20)
