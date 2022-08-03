import cv2
from cv2 import imshow
print(cv2.__version__)
import numpy as np

def color_slicing(image, center, w):
    """
    :param image:
    :param center: b, g, r 取样颜色的bgr值
    :param w: width
    :return:
    """
    r_b, r_g, r_r = center
    out = np.zeros(image.shape, np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            ir,ig,ib = image[x][y]
            # if abs(r_b - a_b) < w / 2 and abs(r_g - a_g) < w / 2 and abs(r_r - a_r) < w / 2:
            #     out[x][y] = image[x][y]
            if (ib>ir)and ib>ig and abs(ib-max(ir,ig))>ib*0.2 :
                out[x][y] = image[x][y]
    return out
img = cv2.imread("3.png")
imshow('img',img)
height, width = img.shape[:2]
print("height : " + str(height) + ', width : ' + str(width))
#裁剪取出图片的下半部分
imgCut = img[ int(height*0.6):height , 0:width ] #(y0,y1,x0,x1)
print(img.shape)
imshow('imgCut',imgCut)
imgRGB = cv2.cvtColor(imgCut, cv2.COLOR_BGR2RGB)

img_slice_red = color_slicing(imgRGB, (75, 0, 0), 0.2549 * 255)
imgBGR = cv2.cvtColor(img_slice_red, cv2.COLOR_RGB2BGR)
imshow('imgRGB',imgRGB)
imshow('red',imgBGR)
cv2.waitKey(0)