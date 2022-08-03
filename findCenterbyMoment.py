##opencv版本：3.4.3
##python版本：3.7.13
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
from cv2 import COLOR_BGR2GRAY
print(cv2.__version__)
import numpy as np
import re
import imutils
#颜色RBG取值,识别颜色种类有："blue","red","yellow","green","purple","orange
color = {
    "blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},
    #"blue": {"color_lower": np.array([100, 102, 0]), "color_upper": np.array([120, 255, 255])},
    "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
    "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
    "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
    "purple": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
    "orange": {"color_lower": np.array([11, 43, 46]), "color_upper": np.array([25, 255, 255])}
         }
#设置识别的颜色为蓝色
color_0 = 'blue'

#读入图像
img = cv2.imread("4.png")

#裁剪取出图片的下半部分
height, width = img.shape[:2]
img = img[ int(height*0.6):height , 0:width ] #(y0,y1,x0,x1)
cv2.imshow("img",img)
imgInfo = img.shape
print(imgInfo)
# 备份一个原图用来进行轮廓标记
imgContour = img.copy()
# #图像转成HSV
# imgHSV = cv2.cvtColor(img,COLOR_BGR2HSV)
# #高斯滤波 第一个参数是读入的图像；第二个参数是高斯核的大小，一般取奇数；第三个参数取标准差为0
imgGauss = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("imgGuass",imgGauss)
# #筛选需要识别的颜色
# SelectImg = cv2.inRange(imgGauss,color[color_0]["color_lower"], color[color_0]["color_upper"])
# cv2.imshow("SelectImg",SelectImg)
# # 腐蚀操作
# # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
# # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
# kernel = np.ones((3, 3), np.uint8)     #核定义
# color_img = cv2.erode(SelectImg, kernel, iterations=2)  #腐蚀除去相关性小的颜色
# cv2.imshow("color_img",color_img)

imgGauss = cv2.cvtColor(imgGauss,COLOR_BGR2GRAY)
#计算轮廓的中心
m = cv2.moments(imgGauss,False)
# print(M)
try:
    cx, cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
except ZeroDivisionError:
        cx, cy = height/2, width/2

# 将光滑的轮廓线折线化
# peri = cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,0.02*peri,True)
# #boundingRect返回的参数：x，y是矩形左上角的点坐标，w，h是宽和高
# x, y, w, h = cv2.boundingRect(approx)
# cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
# cx=int(x+w/2)
# cy=int(y+h/2)

cv2.circle(imgContour, (cx, cy), 4, (0, 0, 255), -1)
#图片中输出中心点位置，putText也可以格式化输出
cv2.putText(imgContour, "x=%d y=%d"%(cx, cy), (cx+20, cy+3),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("imgContour",imgContour)

cv2.waitKey(0)