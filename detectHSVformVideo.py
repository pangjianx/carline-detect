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
#颜色在HSV取值,识别颜色种类有："blue","red","yellow","green","purple","orange
color = {
    #"blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},#原有
    "blue": {"color_lower": np.array([14, 17, 24]), "color_upper": np.array([130, 197, 130])},
    "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
    "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
    "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
    "purple": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
    "orange": {"color_lower": np.array([11, 43, 46]), "color_upper": np.array([25, 255, 255])}
         }
#设置识别的颜色为蓝色
color_0 = 'blue'

#变换后输出的图片分辨率
outWidth = 500
outHeight = 500
video_path      = "1.mp4"
video_save_path = "1_save.mp4"
video_fps = 25.0

capture = cv2.VideoCapture(video_path)
vidio_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vidio_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#保存视频格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#创建视频写入对象
#vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (vidio_width,vidio_height))
vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (outWidth,outHeight))
while(True):
    #读入图像
    ref,img = capture.read()
    if not ref:
        capture.release()
        print("Detection Done,input every key quit!")
        break
    #逆透视变换
    pts1 = np.float32([[190, 353],[450, 353],[4, 422],[637, 422]])

    pts2 = np.float32([[100,100],[400,100],[100,400],[400,400]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (outWidth,outHeight))
    #img = cv2.imread("1.png")
    #裁剪取出图片的下半部分
    height, width = img.shape[:2]
    #img = img[ int(height*0.6):height , 0:width ] #(y0,y1,x0,x1)
    cv2.imshow("img",img)
    
    # 备份一个原图用来进行轮廓标记
    imgContour = img.copy()

    # 腐蚀操作
    # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
    # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
    kernel = np.ones((3, 3), np.uint8)     #核定义
    imgErode = cv2.erode(img, kernel, iterations=1)  #腐蚀除去相关性小的颜色
    cv2.imshow("imgErode",imgErode)
    #膨胀图像
    img = cv2.dilate(imgErode,kernel)
    #显示膨胀后的图像
    cv2.imshow("Dilated Image",img)

    #图像转成HSV
    imgHSV = cv2.cvtColor(img,COLOR_BGR2HSV)
    #高斯滤波 第一个参数是读入的图像；第二个参数是高斯核的大小，一般取奇数；第三个参数取标准差为0
    imgGauss = cv2.GaussianBlur(imgHSV,(5,5),0)
    cv2.imshow("imgGuass",imgGauss)
    #筛选需要识别的颜色
    SelectImg = cv2.inRange(imgGauss,color[color_0]["color_lower"], color[color_0]["color_upper"])
    cv2.imshow("SelectImg",SelectImg)
    

    # 轮廓提取
    #img,contours,hierarchy = cv2.findContours(color_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(SelectImg.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
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
        if area<900 and area>200:
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
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
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