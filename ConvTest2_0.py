import os
import numpy as np
import cv2
import matplot.pyplot as plt
import tensorflow as tf
from tensorflow import contrib
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

def MatchKernerl(dst_img,src_img,FilterSize,BaseRow, BaseCol):
    # dst_img：待检测图像
    # src_img：待检测的印章
    # FilterSize：过滤器的大小。过滤器为印章src_img中某个局部范围，比如四个角中某个角，以（BaseX,BaseY)作为基准点。
    # BaseX：Filter的左上角的X坐标
    # BaseY：Filter的左上角的Y坐标
    # 返回：dst2, Filter2_Norm, dst_image_re, src_img_Norm

    # print('FilterRange:',FilterRange)
    # filters_Source = dst_img[FilterRange[0]:FilterRange[1], FilterRange[2]:FilterRange[3]]#局部模板，比如四个角中某个角
    src_img_Norm = (src_img/128-1)*(-1)#将局部模板转换为【-1,1】之间的数，并让轮廓处的值为1

    Filter = np.zeros([FilterSize, FilterSize])
    Filter[0, 0] = src_img_Norm[BaseRow, BaseCol]#第98行5列为Filter的左上角
    for i in range(Filter.shape[0]):
        for j in range(Filter.shape[1]):
            Filter[i, j] = src_img_Norm[i+BaseRow, j+BaseCol]
    Filter_min = min(Filter.reshape(-1, 1))
    Filter_max = max(Filter.reshape(-1, 1))
    Filter_Norm = (Filter-Filter_min)/(Filter_max - Filter_min)*1.5-0.5#此时min为-0.5，将filter1的像素值转换为[min,1]之间的数,min的取值范围为[-1,-0.1]，min越小，候选的极值点越多。

    dst_image_re = 255 - dst_img  # 让原始图像中轮廓处的灰度值为大数，其他地方为小的数。
    dst = np.zeros(dst_image_re.shape)
    dst = cv2.filter2D(dst_image_re, -1, Filter_Norm)#在周围填充(上下左右分别填充卷积核大小的一半），以保证输出图像尺寸与输入图像尺寸一致。

    # Filter2 = np.zeros([FilterSize, FilterSize])
    # Filter2[0, 0] = src_img_Norm[6, 65]#第98行5列为Filter1的左上角
    # for i in range(Filter2.shape[0]):
    #     for j in range(Filter2.shape[1]):
    #         Filter2[i, j] = src_img_Norm[i+6, j+65]
    # Filter2_min = min(Filter2.reshape(-1, 1))
    # Filter2_max = max(Filter2.reshape(-1, 1))
    # Filter2_Norm = (Filter2-Filter2_min)/(Filter2_max - Filter2_min)*1.5-0.5#此时min为-0.5，将filter1的像素值转换为[min,1]之间的数,min的取值范围为[-1,-0.1]，min越小，候选的极值点越多。
    #
    # dst = np.zeros(dst_image_re.shape)
    # dst = cv2.filter2D(dst_image_re, -1, Filter2_Norm)#在周围填充(上下左右分别填充卷积核大小的一半），以保证输出图像尺寸与输入图像尺寸一致。
    return dst, Filter_Norm, dst_image_re, src_img_Norm

def ShowMaxRsponsePoints(img,dst2, MaxResThreshold, PaddingSize,filtersize):
    # img：需要标注出印章的图像
    # dst2：卷积操作后输出的特征图像
    # MaxResThreshold:响应值的阈值。超过该阈值的点被认为是显著响应点。
    # Padding：卷积操作时所使用的填充大小。
    # 返回：带边框的图像。其中边框用于定位印章。
    pts = np.argwhere(dst2 > MaxResThreshold)  # 找出最大的若干个点
    pts = pts[:, [1, 0]]  # 画边框的时候，是按照（列，行）来定位一个点，而不是（行，列）,因而需要交换行号和列号。
    # print(pts)
    # print(dst2[80, 124], dst2[73:78, 73:78])
    # # 定义四个顶点坐标,顶点个数：4，矩阵变成4*1*2维
    pts = np.array([list(pts)], np.int32).reshape(-1, 1, 2)
    # print(pts.shape)
    # print(pts)
    index = 0
    for index in range(0, pts.shape[0]):
        # print(index)
        # pts1 = pts[index,:].reshape((-1, 1, 2))
        # print(Filter2_Norm.shape)
        # print(pts[index, :])
        basepoint = pts[index, :] - PaddingSize
        basepoint = basepoint.astype("int32")
        # print(basepoint)
        CurPoint = np.concatenate((basepoint, basepoint + [0,filtersize], basepoint + [filtersize, filtersize], basepoint + [filtersize,0]), 0)
        print(CurPoint, dst2.shape)
        print(CurPoint.shape, type(CurPoint))
        # points = np.array([[100, 50], [100, 100], [150, 100], [150, 50]], np.int32)
        # points = CurPoint.reshape(-1, 1, 2)
        # cv2.polylines(dst2,[points],1, (255, 255, 255))
        cv2.polylines(img, [CurPoint], 1, (255, 0, 0))
        index = index + 1
    return img


if __name__ == "__main__":
    # template_image = 'test1.jpg'  # source image(use as a match template)
    detect_image = 'source1.jpg'  # Image to be detected

    img = cv2.imread(detect_image)
    # src_img = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
    dst_img = cv2.imread(detect_image, cv2.IMREAD_GRAYSCALE)
    src_img = dst_img[15:15+120, 60:60+95]

    # dst_image_re = 255-dst_img#让边界处的灰度值为大数，其他地方为小的数。
    # # src_img_Norm = src_img/255
    #
    # filters_Source = dst_img[15:15+120,60:60+95]
    # FilterRange = np.array([15, 15+120, 60, 60+95])


    BaseRow = 0
    BaseCol = 14
    FilterSize = 24
    [dst2, Filter2_Norm, dst_image_re, src_img_Norm] = MatchKernerl(dst_img, src_img, FilterSize,BaseRow, BaseCol)
    PaddingSize = FilterSize * 0.5
    img = ShowMaxRsponsePoints(img, dst2, 253,PaddingSize,FilterSize)


    fig = plt.figure()
    # 展示左上角的框
    sub_img = fig.add_subplot(4,4,1)
    plt.title(u'印章',fontproperties=font_set)
    sub_img.imshow(src_img_Norm, 'gray')#第一个子图
    sub_img = fig.add_subplot(4,4,2)
    plt.title(u'印章的左上角', fontproperties=font_set)
    sub_img.imshow(Filter2_Norm, 'gray')#第二个子图
    sub_img = fig.add_subplot(4,4,3)
    plt.title(u'待检测图像', fontproperties=font_set)
    sub_img.imshow(dst_image_re, 'gray')#第三个子图
    sub_img = fig.add_subplot(4,4,4)
    plt.title(u'标注出匹配区域', fontproperties=font_set)
    sub_img.imshow(img, 'gray')#第四个子图

    BaseRow = 6
    BaseCol = 65
    FilterSize = 24
    [dst2, Filter2_Norm, dst_image_re, src_img_Norm] = MatchKernerl(dst_img, src_img,FilterSize,BaseRow, BaseCol)
    PaddingSize = FilterSize * 0.5
    img = ShowMaxRsponsePoints(img, dst2, 253,PaddingSize,FilterSize)

    # 展示右上角的框
    sub_img = fig.add_subplot(4,4,5)
    plt.title(u'印章',fontproperties=font_set)
    sub_img.imshow(src_img_Norm, 'gray')#第一个子图
    sub_img = fig.add_subplot(4,4,6)
    plt.title(u'印章的左上角', fontproperties=font_set)
    sub_img.imshow(Filter2_Norm, 'gray')#第二个子图
    sub_img = fig.add_subplot(4,4,7)
    plt.title(u'待检测图像', fontproperties=font_set)
    sub_img.imshow(dst_image_re, 'gray')#第三个子图
    sub_img = fig.add_subplot(4,4,8)
    plt.title(u'标注出匹配区域', fontproperties=font_set)
    sub_img.imshow(img, 'gray')#第四个子图

    BaseRow = 90
    BaseCol = 3
    FilterSize = 24
    [dst2, Filter2_Norm, dst_image_re, src_img_Norm] = MatchKernerl(dst_img, src_img,FilterSize,BaseRow, BaseCol)
    PaddingSize = FilterSize * 0.5
    img = ShowMaxRsponsePoints(img, dst2, 253,PaddingSize,FilterSize)

    # 展示左下角的框
    sub_img = fig.add_subplot(4,4,9)
    plt.title(u'印章',fontproperties=font_set)
    sub_img.imshow(src_img_Norm, 'gray')#第一个子图
    sub_img = fig.add_subplot(4,4,10)
    plt.title(u'印章的左上角', fontproperties=font_set)
    sub_img.imshow(Filter2_Norm, 'gray')#第二个子图
    sub_img = fig.add_subplot(4,4,11)
    plt.title(u'待检测图像', fontproperties=font_set)
    sub_img.imshow(dst_image_re, 'gray')#第三个子图
    sub_img = fig.add_subplot(4,4,12)
    plt.title(u'标注出匹配区域', fontproperties=font_set)
    sub_img.imshow(img, 'gray')#第四个子图

    BaseRow = 95
    BaseCol = 50
    FilterSize = 24
    [dst2, Filter2_Norm, dst_image_re, src_img_Norm] = MatchKernerl(dst_img, src_img,FilterSize,BaseRow, BaseCol)
    PaddingSize = FilterSize * 0.5
    img = ShowMaxRsponsePoints(img, dst2, 253,PaddingSize,FilterSize)

    # 展示右下角的框
    sub_img = fig.add_subplot(4,4,13)
    plt.title(u'印章',fontproperties=font_set)
    sub_img.imshow(src_img_Norm, 'gray')#第一个子图
    sub_img = fig.add_subplot(4,4,14)
    plt.title(u'印章的左上角', fontproperties=font_set)
    sub_img.imshow(Filter2_Norm, 'gray')#第二个子图
    sub_img = fig.add_subplot(4,4,15)
    plt.title(u'待检测图像', fontproperties=font_set)
    sub_img.imshow(dst_image_re, 'gray')#第三个子图
    sub_img = fig.add_subplot(4,4,16)
    plt.title(u'标注出匹配区域', fontproperties=font_set)
    sub_img.imshow(img, 'gray')#第四个子图
    plt.show()


    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(src_img_Norm, 'gray')
    # plt.subplot(222)
    # plt.imshow(Filter2_Norm,'gray')
    # plt.subplot(223)
    # plt.imshow(dst_image_re,'gray')
    # plt.subplot(224)
    # # plt.imshow(dst2,'gray')
    # plt.imshow(img)
    # plt.show()


    # dst_image_re = dst_image_re.reshape((1,297,282,1)).astype('float32')
    # # template_image_re_tf = tf.convert_to_tensor(template_image_re)
    # print(template_image_re.shape)
    # # print(type(template_image_re_tf))
    # Filter1_Norm = Filter1_Norm.reshape((24, 24,1, 1)).astype('float32')
    # # Filter1_Norm = tf.convert_to_tensor(Filter1_Norm)
    # dst2 = tf.nn.conv2d(template_image_re, Filter1_Norm, strides=[1, 1, 1, 1], padding='VALID')
    # print(dst2.shape)
    #
    # #
    # # # cv2.imshow('pat', Filter1)
    # #
    # print(type(dst2))
    # sess = tf.Session()
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(src_img_Norm,'gray')
    # plt.subplot(222)
    # plt.imshow(Filter1_Norm[:,:,0,0],'gray')
    # plt.subplot(223)
    # plt.imshow(template_image_re[0,:,:,0],'gray')
    # plt.subplot(224)
    # plt.imshow(dst2[0,:,:,0].eval(session=sess),cmap=plt.cm.gray)
    # plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imshow('src', img)
    #
    # imgInfo = img.shape
    # height = imgInfo[0]
    # width = imgInfo[1]
    # deep = imgInfo[2]
    #
    # # 定义一个旋转矩阵
    # matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 45, 0.7)  # mat rotate 1 center 2 angle 3 缩放系数
    #
    # dst = cv2.warpAffine(img, matRotate, (height, width))
    #
    # cv2.imshow('image', dst)
    # cv2.waitKey(0)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', dst_RGB_img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    ####################
    # src_img_Norm = (filters_Source/128-1)*(-1)#将模板转换为【-1,1】之间的数，并让轮廓处的值为1
    #
    # # Filter1 = np.zeros([12, 12])
    # # Filter1[0, 0] = src_img_Norm[2, 17]#第4行19列为Filter1的左上角
    # # for i in range(Filter1.shape[0]):
    # #     for j in range(Filter1.shape[1]):
    # #         Filter1[i, j] = src_img_Norm[i+2, j+17]
    # # Filter1_min = min(Filter1.reshape(-1, 1))
    # # Filter1_max = max(Filter1.reshape(-1, 1))
    # # Filter1_Norm = (Filter1-Filter1_min)/(Filter1_max - Filter1_min)*1.8-0.8#此时min为-0.8，将filter1的像素值转换为[min,1]之间的数,min的取值范围为[-1,-0.1]，min越小，候选的极值点越多。
    # #
    # # dst2 = np.zeros(dst_image_re.shape)
    # # dst2 = cv2.filter2D(dst_image_re, -1, Filter1_Norm)
    # # print(dst2[20:30,80:90])
    #
    # Filter2 = np.zeros([24, 24])
    # Filter2[0, 0] = src_img_Norm[90, 3]#第98行5列为Filter1的左上角
    # for i in range(Filter2.shape[0]):
    #     for j in range(Filter2.shape[1]):
    #         Filter2[i, j] = src_img_Norm[i+90, j+3]
    # Filter2_min = min(Filter2.reshape(-1, 1))
    # Filter2_max = max(Filter2.reshape(-1, 1))
    # Filter2_Norm = (Filter2-Filter2_min)/(Filter2_max - Filter2_min)*1.5-0.5#此时min为-0.5，将filter1的像素值转换为[min,1]之间的数,min的取值范围为[-1,-0.1]，min越小，候选的极值点越多。
    #
    # dst2 = np.zeros(dst_image_re.shape)
    # dst2 = cv2.filter2D(dst_image_re, -1, Filter2_Norm)#在周围填充(上下左右分别填充卷积核大小的一半），以保证输出图像尺寸与输入图像尺寸一致。
    #
    # Filter2 = np.zeros([24, 24])
    # Filter2[0, 0] = src_img_Norm[6, 65]#第98行5列为Filter1的左上角
    # for i in range(Filter2.shape[0]):
    #     for j in range(Filter2.shape[1]):
    #         Filter2[i, j] = src_img_Norm[i+6, j+65]
    # Filter2_min = min(Filter2.reshape(-1, 1))
    # Filter2_max = max(Filter2.reshape(-1, 1))
    # Filter2_Norm = (Filter2-Filter2_min)/(Filter2_max - Filter2_min)*1.5-0.5#此时min为-0.5，将filter1的像素值转换为[min,1]之间的数,min的取值范围为[-1,-0.1]，min越小，候选的极值点越多。
    #
    # dst2 = np.zeros(dst_image_re.shape)
    # dst2 = cv2.filter2D(dst_image_re, -1, Filter2_Norm)#在周围填充(上下左右分别填充卷积核大小的一半），以保证输出图像尺寸与输入图像尺寸一致。

    # pts = np.argwhere(dst2 > 253)  # 找出最大的若干个点
    # pts = pts[:, [1, 0]]  # 画边框的时候，是按照（列，行）来定位一个点，而不是（行，列）,因而需要交换行号和列号。
    # # print(pts)
    # print(dst2[80, 124], dst2[73:78, 73:78])
    # # # 定义四个顶点坐标,顶点个数：4，矩阵变成4*1*2维
    # pts = np.array([list(pts)], np.int32).reshape(-1, 1, 2)
    # print(pts.shape)
    # print(pts)
    # index = 0
    # for index in range(0, pts.shape[0]):
    #     print(index)
    #     # pts1 = pts[index,:].reshape((-1, 1, 2))
    #     print(Filter2_Norm.shape)
    #     print(pts[index, :])
    #     basepoint = pts[index, :] - np.array(Filter2_Norm.shape)*0.5
    #     basepoint = basepoint.astype("int32")
    #     print(basepoint)
    #     CurPoint = np.concatenate((basepoint, basepoint + [24, 0], basepoint + [24, 24], basepoint + [0, 24]), 0)
    #     print(CurPoint, dst2.shape)
    #     print(CurPoint.shape,type(CurPoint))
    #     # points = np.array([[100, 50], [100, 100], [150, 100], [150, 50]], np.int32)
    #     # points = CurPoint.reshape(-1, 1, 2)
    #     # cv2.polylines(dst2,[points],1, (255, 255, 255))
    #     cv2.polylines(img, [CurPoint], 1, (0, 255, 0))
    #     index = index + 1