##opencv版本：4.6
##python版本：3.9
from tkinter.tix import Select
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import moments
from cv2 import COLOR_BGR2RGB
print(cv2.__version__)
import numpy as np
import re
import time
import imutils
from numba import jit     #使用jit 模块加速for循环   pip install numba or conda install numba
#使用装饰器，jit 加速
@jit         # 就是这么一个简单的改变
#利用RBG颜色空间，保留单个颜色的图像
def colorSelect(image,  w):
    imgOut = np.zeros(image.shape, np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            #读入当前像素点的rgb值
            ir,ig,ib = image[x][y]
            #blue通道的数值大于red和green 并且blue数值比另外通道高出一定的比例w，认为是蓝色像素
            if (ib>ir)and ib>ig and abs(ib-max(ir,ig))>ib*w :
                #填充原有颜色
                #imgOut[x][y] = image[x][y]
                #填充白色
                imgOut[x][y] = 255,255,255
    return imgOut

video_path      = "2.mp4"
video_save_path = "3_save.mp4"
video_fps = 25.0

capture = cv2.VideoCapture(video_path)
vidio_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vidio_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#保存视频格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#创建视频写入对象
vidio_out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (vidio_width,vidio_height-int(vidio_height*0.6)))

while(True):
    #获取一帧图像
    ref,img = capture.read()
    if not ref:
        capture.release()
        print("Detection Done,input every key quit!")
        break

    #裁剪取出图片0.4部分，区域是靠近下半部分
    height, width = img.shape[:2]
    imgCut = img[ int(height*0.6):height , 0:width ] #(y0,y1,x0,x1)
    print('img cut down,img_height:'+str(imgCut.shape[0])+',width:'+str(imgCut.shape[1]))
    cv2.imshow("imgCut",imgCut)
    # 备份一个原图用来进行轮廓标记
    imgContour = imgCut.copy()
    #图像转成RGB,这里转成rgb，colorSelect函数中的img[x][y]的顺序就是rgb，不然会是bgr
    imgRGB = cv2.cvtColor(imgCut,COLOR_BGR2RGB)
    #高斯滤波 第一个参数是读入的图像；第二个参数是高斯核的大小，一般取奇数；第三个参数取标准差为0
    imgRGB = cv2.GaussianBlur(imgRGB,(5,5),0)
    #这里图片已经转为rgb，但是imshow还是以为是bgr，所以图片蓝红对调了
    #cv2.imshow("imgGuass",imgRGB)
    #筛选图片中蓝色像素
    #imgBlue = colorSelect(imgRGB,0.2)
    imgBlue = colorSelect(imgRGB,0.1)
    cv2.imshow("imgBlue_RGB",imgBlue)
    
    # 腐蚀操作 ，实测对图片影响较大，会丢失不少远处信息
    # 因为筛选出来的颜色有些可能在图片的毛刺部分，不是我们想要的结果，会导致误差，因此需要用腐蚀操作进行处理。
    # 先自定义一个核kernel，第一个参数是传入的图像；第二个参数是核；第三个参数是腐蚀的次数。
    # kernel = np.ones((3, 3), np.uint8)     #核定义
    # imgBlue = cv2.erode(imgBlue, kernel, iterations=2)  #腐蚀除去相关性小的颜色
    # cv2.imshow("erode",imgBlue)

    # 轮廓提取
    #findContours传入的图片需要为灰度图片，inRange函数最后输出的也是灰度，所以外面自己筛选的颜色也需要转为灰度
    imgGray = cv2.cvtColor(imgBlue, cv2.COLOR_RGB2GRAY)
    cv2.imshow("imgGray",imgGray)
    contours = cv2.findContours(imgGray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #不同opencv的findContours返回值不一样，利用grab_contours保留最关心的参数
    contours = imutils.grab_contours(contours)
    #print(contours)
    #保存提取的中线在图片中的位置
    station = [(0,0)]
    station.pop()
    #对提取的轮廓进行处理
    for cnt in contours:
        # 计算各个轮廓包围的面积
        area = cv2.contourArea(cnt)
        #print(area)
        # 当面积小于100时进行处理,过滤掉蓝色背景板和噪点
        if area<8000 and area>100:
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

    cv2.imshow("imgContour",imgContour)
    print(station)
    vidio_out.write(imgContour)
    #等待1ms用于显示图像
    cv2.waitKey(1)
print("Save processed video to the path :" + video_save_path)
vidio_out.release()
#cv2.waitKey(0)