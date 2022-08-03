##opencv版本：4.6
##python版本：3.7.13
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
from matplotlib.pyplot import contour
from more_itertools import last
from sympy import print_glsl
print(cv2.__version__)
import numpy as np
import re
import time
import imutils

#设置canny的上下阈值#Canny的高低阈值设置一般为2：1
# canny_low = 55 
# canny_up = 90
canny_low = 23 
canny_up = 70

flagMiss = 0    #丢线的标志位，在两种方法提取的车道线都没有公共点时，丢线标志位置1
selectCx = 0    #当前的车道线x坐标
selectCy = 0    #当前的车道线y坐标
lastCx = 0      #上次车道线的x坐标
lastCy = 0      #上次车道线的y坐标

#丢线容忍值 
# 640 X 480  ，最大偏差是320
#想实现的效果是车道线中点不偏离到一定程度，图像没提取到车道线不认为是丢线，延续上次的车道线坐标
setLossEndure = 200
#设置速度相关参数，用来矫正y轴坐标，车速越大，值越大。
#当前是车速是0.4，speedY是相同箭头中点在两帧之间的y轴之差(182, 390) (177, 392)
speedY= 6
#设置同一帧图片内，最近出两个箭头的y值之差(154, 445) (193, 406)
arryY = 39
video_path      = "31.mp4"
video_save_path = "31_threshold_save.mp4"
video_save_path2 = "31_canny_save.mp4"
video_save_path3 = '31_and_save.mp4'
video_fps = 25.0

capture = cv2.VideoCapture(video_path)
vidio_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vidio_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#保存视频格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#创建视频写入对象
vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (144,vidio_height))
vidio_out2 = cv2.VideoWriter(video_save_path2, fourcc, video_fps, (144,vidio_height))
vidio_out3 = cv2.VideoWriter(video_save_path3, fourcc, video_fps, (144,vidio_height))
def findAnd(listA,listB): ##[ (234,432), (234,466) ]
    #求交集
    retF = list(set(listA)&set(listB))
    print ("intersection is: ",retF)
    return retF
#双层循环遍历寻找交集
def findAnd2(listA,listB):
    retList = []
    for i in listA:
        for j in listB:
            #遍历B，查看A是否在B中
            if abs(i[0]-j[0])<4:
                if abs(i[1]-j[1]<4):
                    retList.append(i)
                    break
    print ("intersection is: ",retList)
    return retList
 
while(True):
    #读入图像
    ref,img = capture.read()
    if not ref:
        capture.release()
        print("Detection Done,input every key quit!")
        break
    
    #裁剪取出图片的下半部分
    img = img[336:480,0:640] #(y0,y1,x0,x1)
    cv2.imshow("img",img)
    #打印图片的分辨率
    print('img cut size width:%d heigh:%d'%(img.shape[0],img.shape[1]))
    
    # 备份一个原图用来进行轮廓标记
    imgContour = img.copy()
    imgContour2 = img.copy()
    imgContour3 = img.copy()
    #Canny提取轮廓容易受到噪声影响，用高斯滤波去除高频噪声 效果并不好
    #img = cv2.GaussianBlur(img,(5,5),0)
    #中值滤波
    img = cv2.medianBlur(img, 5)
    # 腐蚀操作
    # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
    # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
    kernel = np.ones((3, 3), np.uint8)     #核定义
    img = cv2.erode(img, kernel, iterations=1)  #腐蚀除去相关性小的颜色
    cv2.imshow("ErodeImg",img)
    #膨胀图像
    img = cv2.dilate(img,kernel)
    #显示膨胀后的图像
    cv2.imshow("DilatedImg",img)

    #图像再转为灰度图
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #筛选需要识别的颜色
    #ret,SelectImg=cv2.threshold(imgGRAY,thresh,255, cv2.THRESH_BINARY)
    #cv2.imshow('binImg',SelectImg)
    #把窗口设置足够大以后但不能超过图像大小，得到的结果就与全局阈值相同
    #窗口大小使用的为11，当窗口越小的时候，得到的图像越细。
    #中值自适应滤波和高斯自适应滤波效果没有具体对比，这里选择了高斯
    #SelectImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
    SelectImg = cv2.adaptiveThreshold(imgGRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imshow('binImg',SelectImg)

    imgCanny = cv2.Canny(imgGRAY, canny_low, canny_up)#Canny的高低阈值设置一般为2：1
    cv2.imshow("imgCanny",imgCanny)

    
    # 轮廓提取
    #img,contours,hierarchy = cv2.findContours(color_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contoursThreshold = cv2.findContours(SelectImg.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contourCanny = cv2.findContours(imgCanny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #为了兼容不同opencv的版本，grab_contours是取不同版本的cnts
    contoursThreshold = imutils.grab_contours(contoursThreshold)
    contourCanny = imutils.grab_contours(contourCanny)
    #print(contours)
    #保存提取的中线在图片中的位置
    station = [(0,0)]
    station2 = [(0,0)]
    station.pop()
    station2.pop()
    #对提取的轮廓进行处理
    for cnt in contoursThreshold:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        if area<1300 and area>200:
            print(area)
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
            cx = int(x+w/2)
            cy = int(y+h/2)
            cv2.circle(imgContour, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
            #图片中输出中心点位置，putText也可以格式化输出
            cv2.putText(imgContour, "x=%d y=%d"%(int(x+w/2), int(y+h/2)), (int(x+w/2)+20, int(y+h/2)+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #只保存图片下半部分的点
            # if(cy>vidio_height*0.7):
            #     station.append( (int(x+w/2), int(y+h/2)) )
            station.append( (int(x+w/2), int(y+h/2)) )
    #对提取的轮廓进行处理
    for cnt in contourCanny:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        if area<8000 and area>200:
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
            cx = int(x+w/2)
            cy = int(y+h/2)
            cv2.circle(imgContour2, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
            #图片中输出中心点位置，putText也可以格式化输出
            cv2.putText(imgContour2, "x=%d y=%d"%(int(x+w/2), int(y+h/2)), (int(x+w/2)+20, int(y+h/2)+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # #只保存图片下半部分的点
            # if(cy>vidio_height*0.7):
            #     station2.append( (int(x+w/2), int(y+h/2)) )
            station2.append( (int(x+w/2), int(y+h/2)) )
    cv2.imshow("imgThresholdFind",imgContour)
    cv2.imshow("imgCannyFind",imgContour2)
    print("Threshold station")
    print(station)
    # print(type(station))
    # print(type(station[0]))
    print("Canny station")
    print(station2)
    
    #保存上一次的车道线坐标
    lastCx = selectCx      
    lastCy = selectCy  
    #求交集
    s = findAnd2(station,station2)
    #找到车道线
    if len(s)>0:
        #给车道线打点
        for i in s:
            cv2.circle(imgContour3, (i[0],i[1]), 7, (255, 255, 255), -1)
            cv2.putText(imgContour3, "x=%d y=%d"%(i[0],i[1]), (i[0]+20, i[1]+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #更新坐标数值
        flagMiss = 0
        #s[0][0] ,s[0][1]取第一个点
        #s[1][0], s[1][1]取第二个点
        selectCx = s[0][0]    
        selectCy = s[0][1]    
            # print(i)#(248,403)
            # print(type(s))#list
            # print(type(i))#tuple
            # print(i[0])
        #检查上次车道线是否突变,如果突变认为是错误的
        # if selectCx - lastCx >100:
        #     selectCx = lastCx
        #     selectCy = lastCy
    #未找到车道线
    else:
        if lastCx < (320 - setLossEndure): #上次的位置是在极左边
            flagMiss = 1#丢线标志
            selectCx = lastCx
            selectCy = lastCy
        else: 
            if lastCx > (320 + setLossEndure):#上次的位置是在极右边
                flagMiss = 1#丢线标志
                selectCx =lastCx
                selectCy =lastCy
            else: #处于中间，认为只是突然的丢帧,丢线标志不置位
                flagMiss = 0
                selectCx = lastCx
                selectCy = lastCy

    cv2.circle(imgContour3, (selectCx,selectCy), 7, (255, 0, 0), -1)
    cv2.putText(imgContour3, "x=%d y=%d"%(selectCx,selectCy), (selectCx+20, selectCy+3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print("flagMiss=%d"%flagMiss)
    cv2.imshow("imgFind",imgContour3)

    vidio_out.write(imgContour)
    vidio_out2.write(imgContour2)
    vidio_out3.write(imgContour3)
    #等待1ms用于显示图像
    #cv2.waitKey(1)
    cv2.waitKey(0)
    k = cv2.waitKey(1)&0xFF
    if k == 27: #esc exit
        break
print("Save processed video to the path :" + video_save_path)
vidio_out.release()
#cv2.waitKey(0)