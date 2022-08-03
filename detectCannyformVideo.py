##opencv版本：4.6
##python版本：3.7.13
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
print(cv2.__version__)
import numpy as np
import re
import time
import imutils

#设置canny的上下阈值#Canny的高低阈值设置一般为2：1
canny_low = 70
canny_up = 200
video_path      = "1.mp4"
video_save_path = "1_save.mp4"
video_fps = 25.0

capture = cv2.VideoCapture(video_path)
vidio_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vidio_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#保存视频格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#创建视频写入对象
vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (vidio_width,vidio_height))

while(True):

    #读入图像
    ref,img = capture.read()
    if not ref:
        capture.release()
        print("Detection Done,input every key quit!")
        break

    #裁剪取出图片的下半部分
    #img = img[500:1080,0:1920] #(y0,y1,x0,x1)
    cv2.imshow("img",img)
    height, width = img.shape[:2]
    # 备份一个原图用来进行轮廓标记
    imgContour = img.copy()
    
    #高斯滤波 第一个参数是读入的图像；第二个参数是高斯核的大小，一般取奇数；第三个参数取标准差为0
    imgGauss = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow("imgGuass",imgGauss)
    
    #图像转为灰度图
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGRAY, canny_low, canny_up)#Canny的高低阈值设置一般为2：1
    cv2.imshow("imgCanny",imgCanny)
    # # 腐蚀操作
    # # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
    # # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
    # kernel = np.ones((3, 3), np.uint8)     #核定义
    # imgErode = cv2.erode(imgCanny, kernel, iterations=1)  #腐蚀除去相关性小的颜色
    # cv2.imshow("imgErode",imgErode)
    # #膨胀图像
    # dilated = cv2.dilate(imgErode,kernel)
    # #显示膨胀后的图像
    # cv2.imshow("Dilated Image",dilated)

    # 轮廓提取
    #img,contours,hierarchy = cv2.findContours(color_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(imgCanny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #为了兼容不同opencv的版本，grab_contours是取不同版本的cnts
    contours = imutils.grab_contours(contours)
    #print(contours)
    #保存提取的中线在图片中的位置
    station = [(0,0)]
    station.pop()
    #对提取的轮廓进行处理
    for cnt in contours:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        # 画轮廓线（绿色）
        cv2.drawContours(imgContour, cnt, -1, (0,255,  0), 2,8)
        print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        if area<8000 and area>10:
            # 画轮廓线（绿色）
            cv2.drawContours(imgContour, cnt, -1, (0,255,  0), 2,8)
        
            #计算轮廓的中心
            # m = cv2.moments(cnt)
            # # print(M)
            # try:
            #     cx, cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
            # except ZeroDivisionError:
            #         cx, cy = height/2, width/2

            # 将光滑的轮廓线折线化
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #boundingRect返回的参数：x，y是矩形左上角的点坐标，w，h是宽和高
            x, y, w, h = cv2.boundingRect(approx)
            #画方框
            #cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cx=int(x+w/2)
            cy=int(y+h/2)

            cv2.circle(imgContour, (cx, cy), 4, (0, 0, 255), -1)
            #图片中输出中心点位置，putText也可以格式化输出
            cv2.putText(imgContour, "x=%d y=%d"%(cx, cy), (cx+20, cy+3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            station.append( (cx, cy) )
    cv2.imshow("imgContour",imgContour)
    print(station)
    vidio_out.write(imgContour)
    #等待1ms用于显示图像
    cv2.waitKey(1)
    cv2.waitKey(0)
print("Save processed video to the path :" + video_save_path)
vidio_out.release()
#cv2.waitKey(0)