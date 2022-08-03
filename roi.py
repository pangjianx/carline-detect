import cv2


src = cv2.imread(".\image\image00.png",1)
proimage0 = src.copy()#复制原图


roi = cv2.selectROI(windowName="roi", img=src, showCrosshair=False, fromCenter=False)#感兴趣区域ROI
x, y, w, h = roi 
cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)#在图像绘制区域
cv2.imshow("roi", src)

#进行裁剪
ImageROI=proimage0[y:y+h,x:x+w]
cv2.imshow("ImageROI", ImageROI)

cv2.waitKey(0)
cv2.destroyAllWindows()