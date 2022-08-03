import cv2
import numpy as np
 
# 图像1
img = cv2.imread("11.png")
pts1 = np.float32([[321, 250],[408, 250],[241, 393],[477, 393]])
ROI_HEIGHT = 30000
ROI_WIDTH = 3750

# 设定逆透视图的宽度
IPM_WIDTH = 500
N = 5

# 保证逆透视图的宽度大概为N个车头宽
sacale=(IPM_WIDTH/N)/ROI_WIDTH  #0.0266666666666667
IPM_HEIGHT=ROI_HEIGHT*sacale #800

# pts2 = np.float32([[IPM_WIDTH/2-IPM_WIDTH/(2*N), 0], #200,0
#                    [IPM_WIDTH/2+IPM_WIDTH/(2*N), 0], #300,0
#                    [IPM_WIDTH/2-IPM_WIDTH/(2*N), IPM_HEIGHT], #200,800
#                    [IPM_WIDTH/2+IPM_WIDTH/(2*N), IPM_HEIGHT]]) #300,800
pts2 = np.float32([[200,300],[300,300],[200,800],[300,800]])
print(IPM_HEIGHT,IPM_WIDTH)

matrix = cv2.getPerspectiveTransform(pts1, pts2)
#output = cv2.warpPerspective(img, matrix, (int(IPM_WIDTH),int(IPM_HEIGHT+50))) #500,850
output = cv2.warpPerspective(img, matrix, (600,900))
for i in range(0, 4):
    cv2.circle(img, (int(pts1[i][0]), int(pts1[i][1])), 4, (0, 0, 255), -1)

for i in range(0,4):
    cv2.circle(output, (int(pts2[i][0]), int(pts2[i][1])),4, (0, 0, 255), -1)

# p1 = (0, 250)
# p2 = (img.shape[1], img.shape[0]-100)
# point_color = (255, 0, 0)
# cv2.rectangle(img, p1, p2, point_color, 2)

cv2.imshow("src image", img)
cv2.imshow("output image", output)
cv2.waitKey(0)