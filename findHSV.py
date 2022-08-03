import cv2
import numpy as np

def callback(x):
    pass

#通过Opencv读取图片信息
#img = cv2.imread('.\image\image00.png')
img = cv2.imread ("1.png")
outWidth = 500
outHeight = 500

# pts1 = np.float32([[190, 353],[450, 353],[4, 422],[637, 422]])#bd2

# pts2 = np.float32([[100,100],[400,100],[100,400],[400,400]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img = cv2.warpPerspective(img, matrix, (outWidth,outHeight))
#分辨率比较高重新定义分辨率
#img = cv2.resize(img, None, fx=0.6, fy=0.4, interpolation=cv2.INTER_CUBIC)
print(img.shape)
rows,cols,channels = img.shape
cv2.imshow("origin", img)
cv2.namedWindow('img2',1)
cv2.resizeWindow("img2", 500, 300) #创建一个500*500大小的窗口

# 创建6个滑条用来操作HSV3个分量的上下截取界限
# 函数的第一个参数是滑动条的名字，
# 第二个参数是滑动条被放置的窗口的名字，
# 第三个参数是滑动条默认值，
# 第四个参数是滑动条的最大值，
# 第五个参数是回调函数，每次滑动都会调用回调函数。
cv2.createTrackbar('Hlow','img2',0,255,callback)
cv2.createTrackbar('Hup','img2',180,255,callback)
cv2.createTrackbar('Slow','img2',0,255,callback)
cv2.createTrackbar('Sup','img2',255,255,callback)
cv2.createTrackbar('Vlow','img2',0,255,callback)
cv2.createTrackbar('Vup','img2',255,255,callback)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while(1):
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    #将制定像素点的数据设置为0, 要注意的是这三个参数对应的值是Blue, Green, Red。
    hlow = cv2.getTrackbarPos('Hlow', 'img2')
    hup = cv2.getTrackbarPos('Hup', 'img2')
    slow = cv2.getTrackbarPos('Slow', 'img2')
    sup = cv2.getTrackbarPos('Sup', 'img2')
    vlow = cv2.getTrackbarPos('Vlow', 'img2')
    vup = cv2.getTrackbarPos('Vup', 'img2')
    lower_red = np.array([hlow, slow, vlow])
    upper_red = np.array([hup, sup, vup])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    img2 = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("img2,esc quit", img2)
    k = cv2.waitKey(1)&0xFF
    if k == 27: #esc exit
        break
#cv2.waitKey(0)
cv2.destroyAllWindows()
