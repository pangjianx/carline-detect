##opencv版本：3.4.3
##python版本：3.7.13
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
print(cv2.__version__)
import numpy as np
import re
import imutils
def callback(x):
    pass

#读入图像
img = cv2.imread("4.png")

cv2.imshow("origin", img)
#高斯滤波 第一个参数是读入的图像；第二个参数是高斯核的大小，一般取奇数；第三个参数取标准差为0
# img = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("imgGuass",img)

#img = cv2.medianBlur(img, 5)
# 腐蚀操作
# 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
# 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
# canny使用的时候提取的本来就是1个像素的边缘，所以腐蚀卷积核是3直接没了
kernel = np.ones((7, 7), np.uint8)     #核定义
imgErode = cv2.erode(img, kernel, iterations=1)  #腐蚀除去相关性小的颜色
cv2.imshow("imgErode",imgErode)
#膨胀图像
dilated = cv2.dilate(imgErode,kernel)
#显示膨胀后的图像
cv2.imshow("Dilated Image",dilated)
gray_image = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('img2')
cv2.resizeWindow("img2", 500, 100) #创建一个500*500大小的窗口
# 创建6个滑条用来操作HSV3个分量的上下截取界限
# 函数的第一个参数是滑动条的名字，
# 第二个参数是滑动条被放置的窗口的名字，
# 第三个参数是滑动条默认值，
# 第四个参数是滑动条的最大值，
# 第五个参数是回调函数，每次滑动都会调用回调函数。
cv2.createTrackbar('Cannylow','img2',0,1000,callback)
cv2.createTrackbar('Cannyup','img2',180,1000,callback)
while(1): 
    low = cv2.getTrackbarPos('Cannylow', 'img2')
    up = cv2.getTrackbarPos('Cannyup', 'img2')
    canny_image = cv2.Canny(gray_image, low, up)#Canny的高低阈值设置一般为2：1
    cv2.imshow("canny_image", canny_image)

    
    
    
    contourCanny = cv2.findContours(canny_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contourCanny = imutils.grab_contours(contourCanny)
    imgCounter2 = img.copy()
    for cnt in contourCanny:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        if area<8000 and area>10:
            # 画轮廓线（绿色）
            cv2.drawContours(imgCounter2, cnt, -1, (0,0,  0), 2,8)
            
            # #计算轮廓的中心
            # M = cv2.moments(cnt)
            # print(M)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # cv2.circle(imgContour, (cX, cY), 7, (255, 255, 255), -1)
            # 将光滑的轮廓线折线化
            peri2 = cv2.arcLength(cnt,True)
            approx2 = cv2.approxPolyDP(cnt,0.02*peri2,True)
            #boundingRect返回的参数：x，y是矩形左上角的点坐标，w，h是宽和高
            x, y, w, h = cv2.boundingRect(approx2)
            cv2.rectangle(imgCounter2,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(imgCounter2, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
            #图片中输出中心点位置，putText也可以格式化输出
            cv2.putText(imgCounter2, "x=%d y=%d"%(int(x+w/2), int(y+h/2)), (int(x+w/2)+20, int(y+h/2)+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("imgContour2",imgCounter2)       
    #cv2.waitKey(0)
    k = cv2.waitKey(1)&0xFF
    if k == 27: #esc exit
        break
cv2.destroyAllWindows()