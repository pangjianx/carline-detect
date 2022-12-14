# 开发日志

RGB每个通道范围是0-255 HSV实际范围H(0-360)S(0-100)V(0-100) opencv中存储为H(0-180)S(0-255)V(0-255)
HSV转opencv的HSV，cv2HSV H/2 S/100*256 V/100*256

场景1：
白色赛道背景1：RGB(141,181,160)对应的HSV(149,22,71);白色赛道背景2：RGB(146,183,166)HSV(152,20,72)

灰色地板：RGB(134,142,135)HSV(127,6,56)

透明窗户：RGB(248,255,255)HSV(180,3,100);RGB(255,255,255)HSV(0,0,100)

红色消防栓：RGB(83,39,27)HSV(13,67,33)
红色赛道弧线：RGB(82,36,40)HSV(355,56,32);RGB(81,26,26)HSV(0,68,32);RGB(95,72,78)HSV(344,24,37);

赛道蓝色弧线：RGB(73,112,142)HSV(206,49,56);RGB(35,50,93)HSV(224,62,36);RGB(43,48,87)HSV(233,51,34)
蓝色背景围板：RGB(0,15,46)HSV(220,100,18);RGB(0,12,71)HSV(230,100,28);  RGB(0,1,32)HSV(238,100,13);
赛道中线：   RGB(15,41,89)HSV(219,83,35);RGB(60,106,143)HSV(207,58,56);RGB(95,151,185)HSV(203,49,73);
             RGB(0,32,80)HSV(216,100,31);RGB(64,110,138)HSV(203,54,54);RGB(76,122,152)HSV(204,50,60);

场景2：
赛道灰色背景：RGB(115,126,127)HSV(185,9,50);RGB(112,118,120)HSV(195,7,47)
灰色地板：RGB(154,154,152)HSV(60,1,60) RGB(121,132,124)HSV(136,8,52) 
黑色窗户：RGB(51,61,69)HSV(207,26,27) RGB(45,46,52)HSV(231,13,20)RGB(34,46,53)HSV(202,36,21) 特征在于rgb三个通道数值很接近
红色赛道弧线：RGB(226,139,69)HSV(207,26,27)

## 7.7

结合canny和hsv共同提取视频中的箭头车道线
canny提取有噪点，hsv提取也有噪点，但噪点不相同
分别用两种方式提取出轮廓的中心点，认为共同的中心点才是可信度高的点
结果：失败
提取点时，m提取和矩形框提取的点不一样，canny和hsv用同种方法计算的中点也不一样

## 7.8

开始做逆透视变换，把视角转换为俯视图
这样每个车道线的箭头面积大小应该一致，用面积过滤噪点，面积范围可以收缩到很小，能够实现用面积过滤掉噪点
逆透视变换需要的知道一个矩形框在平视图中的坐标，和实际世界的长宽，这里有一点在矩形内的像素点投影后误差小，矩形外误差大。因此要矩形要尽量大，包括住主要识别的部分。
先利用cv2.getPerspectiveTransform函数得到变换矩阵(matrix) ,此函数需要两组坐标，一组为图片中矩形的四个角所在的像素点位置，另一组为这个矩形的俯视图视角占的像素点坐标，长宽的比例需要和实际世界中的一样。
再利用cv2.warpPerspective进行透视变换，需要传入变换矩阵matrix，和最终的输出图片分辨率
注意：两组矩形坐标的顺序都是左上，右上，左下，右下
参考：【opencv实践】仿射变换和透视变换 C++ 版本 <https://mp.weixin.qq.com/s/oQePv0EvapVCJM83yvOtIw>
python 版本 扑克牌做逆透视变换  <https://www.bilibili.com/video/BV17h411X7JB?spm_id_from=333.337.search-card.all.click&vd_source=4771f2f37d5d91bb3d6c478e4e198809>

## 7.9

在hsv提取中加入逆透视变换，查看提取的准确性s
提取后半部分不太准确

## 7.18

利用ps提升了亮度和对比度，相较于原图，提升亮度和对比对没有使二值化效果变好
使用二值化，尝试自适应二值化，自适应二值化的效果比全局二值化好，自适应二值化的效果更像是另一种canny提取边缘
自适应二值化和canny一起提取实现没有噪点的检出车道线
提取车道线后的处理思路
在处理一帧图像之前保存上次的车道线提取值
做提取处理后，如果没有提取到车道线
    查看上次的车道线位置，如果上次车道线位置很偏
    认为是车身偏差太大，可以使用上次的偏差值
提取到车道线后
    更新新的位置
## 7.27处理选点不稳定
### 优化：重构图像尺寸只计算下面0.3部分
640 480  0.3*480=144 0.7*480=336
### 对二值化的点进行距离判断滤去噪点
先利用共同点为起点，在x一定范围和y一定范围内，认为是相邻点
[(363, 2), (77, 98), (106, 86), (170, 60), (234, 44), (435, 84), (143, 27), (337, 22)]