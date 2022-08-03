import cv2
import numpy as np

def callback(x):
    pass

# 读取图像
img = cv2.imread("1.png")
#分辨率比较高重新定义分辨率
#img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
print(img.shape)
# 用来进行轮廓标记的图像
imgContour = img.copy()

# 灰度转换、模糊、边缘提取
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("imgGray",imgGray)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
cv2.imshow("imgBlur",imgBlur)
#开窗放置滑条
cv2.namedWindow('adjust',1)
cv2.resizeWindow("adjust", 500, 500) #创建一个500*500大小的窗口
#创建2个滑条用来操作canny的上下截取界限 
# 函数的第一个参数是滑动条的名字，
# 第二个参数是滑动条被放置的窗口的名字，
# 第三个参数是滑动条默认值，
# 第四个参数是滑动条的最大值，
# 第五个参数是回调函数，每次滑动都会调用回调函数。
cv2.createTrackbar('CannyLow','adjust',54,500,callback)
cv2.createTrackbar('CannyUp','adjust',394,500,callback)
while(1):
    cannylow = cv2.getTrackbarPos('CannyLow', 'adjust')
    cannyup = cv2.getTrackbarPos('CannyUp', 'adjust')
    imgCanny = cv2.Canny(imgBlur,cannylow,cannyup)
    #print(imgCanny.shape)

    # 轮廓提取
    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # 对提取的各个轮廓进行遍历
    for cnt in contours:        
            # 计算各个轮廓包围的面积
            area = cv2.contourArea(cnt)
            print(area)
            
            # 当面积大于500时进行处理
            if area>500:
                
                # 画轮廓线（蓝色）
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                
                # 将光滑的轮廓线折线化
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            
                # 画近似折线 （红色）
                cv2.polylines(imgContour, [approx], True, (0, 0, 255), 2)
                
                # 利用一个矩形将检测目标圈起来（绿色）
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
                
                # 根据近似折线段的数目判断目标的形状
                objCor = len(approx)
                
                # 三条线段为三角形
                if objCor ==3: objectType ="Tri"
                
                # 四条线段时，根据目标包围矩形的宽高比判断是 长方形还是正方形
                elif objCor == 4:
                    aspRatio = w/float(h)
                    if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                    else:objectType="Rectangle"
                
                # 四条以上线段时为圆形
                elif objCor>4: objectType= "Circles"
                else:objectType="None"
                
                # 将判断的形状标注在图像上
                cv2.putText(imgContour,objectType,
                            (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                            (0,0,0),2)
    #cv2.imshow("adjust,esc quit", img2)
    k = cv2.waitKey(1)&0xFF
    if k == 27: #esc exit
        break
    cv2.imshow("img",img)
    cv2.imshow("edge",imgCanny)
    cv2.imshow("shape",imgContour)
