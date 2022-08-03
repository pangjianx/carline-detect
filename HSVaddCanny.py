##opencv版本：4.6
##python版本：3.7.13
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
from matplotlib.pyplot import contour
from sympy import print_glsl
print(cv2.__version__)
import numpy as np
import re
import time
import imutils
#颜色在HSV取值,识别颜色种类有："blue","red","yellow","green","purple","orange
color = {
    #"blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},
    "blue": {"color_lower": np.array([27, 60, 0]), "color_upper": np.array([153, 230, 155])},
    "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
    "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
    "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
    "purple": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
    "orange": {"color_lower": np.array([11, 43, 46]), "color_upper": np.array([25, 255, 255])}
         }
#设置识别的颜色为蓝色
color_0 = 'blue'
#设置canny的上下阈值#Canny的高低阈值设置一般为2：1
canny_low = 38
canny_up = 144

video_path      = "2.mp4"
video_save_path = "2_hsv_save.mp4"
video_save_path2 = "2_canny_save.mp4"
video_fps = 25.0

capture = cv2.VideoCapture(video_path)
vidio_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vidio_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#保存视频格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#创建视频写入对象
vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (vidio_width,vidio_height))
vidio_out2 = cv2.VideoWriter(video_save_path2, fourcc, video_fps, (vidio_width,vidio_height))

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

    # 备份一个原图用来进行轮廓标记
    imgContour = img.copy()
    imgContour2 = img.copy()

    #Canny提取轮廓容易受到噪声影响，用高斯滤波去除高频噪声
    img = cv2.GaussianBlur(img,(5,5),0)

    # 腐蚀操作
    # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
    # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
    kernel = np.ones((5, 5), np.uint8)     #核定义
    img = cv2.erode(img, kernel, iterations=1)  #腐蚀除去相关性小的颜色
    cv2.imshow("ErodeImg",img)
    #膨胀图像
    img = cv2.dilate(img,kernel)
    #显示膨胀后的图像
    cv2.imshow("DilatedImg",img)

    #图像转成HSV
    imgHSV = cv2.cvtColor(img,COLOR_BGR2HSV)
    #图像再转为灰度图
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #筛选需要识别的颜色
    imgCanny = cv2.Canny(imgGRAY, canny_low, canny_up)#Canny的高低阈值设置一般为2：1
    cv2.imshow("imgCanny",imgCanny)

    SelectImg = cv2.inRange(imgHSV,color[color_0]["color_lower"], color[color_0]["color_upper"])
    cv2.imshow("imgHSV",SelectImg)
    
    # 轮廓提取
    #img,contours,hierarchy = cv2.findContours(color_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contoursHsv = cv2.findContours(SelectImg.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contourCanny = cv2.findContours(imgCanny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #为了兼容不同opencv的版本，grab_contours是取不同版本的cnts
    contoursHsv = imutils.grab_contours(contoursHsv)
    contourCanny = imutils.grab_contours(contourCanny)
    #print(contours)
    #保存提取的中线在图片中的位置
    station = [(0,0)]
    station2 = [(0,0)]
    station.pop()
    station2.pop()
    #对提取的轮廓进行处理
    for cnt in contoursHsv:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        # if area<8000 and area>200:
        # 画轮廓线（绿色）
        cv2.drawContours(imgContour, cnt, -1, (0,0,  0), 2,8)
        
        # #计算轮廓的中心
        # M = cv2.moments(cnt)
        # print(M)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # cv2.circle(imgContour, (cX, cY), 7, (255, 255, 255), -1)
        # 将光滑的轮廓线折线化
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.02*peri,True)
        #boundingRect返回的参数：x，y是矩形左上角的点坐标，w，h是宽和高
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(imgContour, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
        #图片中输出中心点位置，putText也可以格式化输出
        cv2.putText(imgContour, "x=%d y=%d"%(int(x+w/2), int(y+h/2)), (int(x+w/2)+20, int(y+h/2)+3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        station.append( (int(x+w/2), int(y+h/2)) )
     #对提取的轮廓进行处理
    for cnt in contourCanny:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        # if area<8000 and area>200:
        # 画轮廓线（绿色）
        cv2.drawContours(imgContour2, cnt, -1, (0,0,  0), 2,8)
        
        # #计算轮廓的中心
        # M = cv2.moments(cnt)
        # print(M)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # cv2.circle(imgContour, (cX, cY), 7, (255, 255, 255), -1)
        # 将光滑的轮廓线折线化
        peri2 = cv2.arcLength(cnt,True)
        approx2 = cv2.approxPolyDP(cnt,0.02*peri,True)
        #boundingRect返回的参数：x，y是矩形左上角的点坐标，w，h是宽和高
        x, y, w, h = cv2.boundingRect(approx2)
        cv2.rectangle(imgContour2,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(imgContour2, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
        #图片中输出中心点位置，putText也可以格式化输出
        cv2.putText(imgContour2, "x=%d y=%d"%(int(x+w/2), int(y+h/2)), (int(x+w/2)+20, int(y+h/2)+3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        station2.append( (int(x+w/2), int(y+h/2)) )
    cv2.imshow("imgHSVFind",imgContour)
    cv2.imshow("imgCannyFind",imgContour2)
    print("HSV station")
    print(station)
    # print(type(station))
    # print(type(station[0]))
    print("Canny station")
    print(station2)
    vidio_out.write(imgContour)
    vidio_out2.write(imgContour2)
    #等待1ms用于显示图像
    # cv2.waitKey(1)
    cv2.waitKey(0)
    k = cv2.waitKey(1)&0xFF
    if k == 27: #esc exit
        break
print("Save processed video to the path :" + video_save_path)
vidio_out.release()
#cv2.waitKey(0)