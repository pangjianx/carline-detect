import cv2
import numpy as np
from matplotlib import pyplot as plt
#img = cv2.imread('.\image\image00.png', 0)
img = cv2.imread("2.png",0)
img = cv2.medianBlur(img, 5)
ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#把窗口设置足够大以后但不能超过图像大小，得到的结果就与全局阈值相同
#窗口大小使用的为11，当窗口越小的时候，得到的图像越细。
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
titles = ['Original Image', 'Global Thresholding v=127','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

