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
from utils import LOAD_GRAY_SCALE
import cv2
import  numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    image = cv2.imread(img_path, rep - 1)/255.0

    def on_trackbar(gamma: int):
        gamma_img = np.power(image, gamma * 0.01)
        cv2.imshow("Gamma Correction", gamma_img)

    title_window= 'Gamma Correction'
    gamma_slider_max = 200
    trackbarName = 'Gamma'

    cv2.namedWindow(title_window)
    cv2.createTrackbar(trackbarName, title_window, 0, gamma_slider_max, on_trackbar)

    on_trackbar(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main():
    gammaDisplay('beach.jpg', 2)


if __name__ == '__main__':
    main()
