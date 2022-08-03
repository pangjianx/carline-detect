import cv2
import numpy as np
import imutils
img = cv2.imread("1.png")
b, g, r = cv2.split(img)
row,column = b.shape[0],b.shape[1]
for i in range(row-150):
    for j in range(column):
        b[i,j] = 0
        g[i, j] = 0
        r[i, j] = 0
result = cv2.merge([b,g,r])
# 2.提取图片中的蓝色部分
hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
low_hsv = np.array([100,43,46])
high_hsv = np.array([124,255,255])
mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
cv2.imshow("find_yellow",mask)
# cv2.imwrite("result.png",mask)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# 3.腐蚀操作，去掉部分光线带来的影响
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  #设置kernel卷积核为 3 * 3 正方形，8位uchar型，全1结构元素
# mask = cv2.erode(mask, kernel,25)
# cv2.imshow("morphology", mask)
# 遍历轮廓集
for c in cnts:
    # 计算轮廓区域的图像矩。 在计算机视觉和图像处理中，图像矩通常用于表征图像中对象的形状。这些力矩捕获了形状的基本统计特性，包括对象的面积，质心（即，对象的中心（x，y）坐标），方向以及其他所需的特性。
    M = cv2.moments(c)

    if M["m00"]==0:
        break
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print("中心点坐标：","x=",cX,"y=",cY)
    # 在图像上绘制轮廓及中心
    cv2.drawContours(mask, [c], -1, (0, 255, 0), 2)
    cv2.circle(mask, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(mask, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # 展示图像
    cv2.imshow("Image", mask)
cv2.waitKey(0)