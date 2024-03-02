# -*- coding: UTF-8 -*-
import  cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import solve

def neighbor4_4coef(mat, i, j, list):
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    x = ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2] + ls[3] * list[3]
    return round(x,3)

def point_loc_type(mat, i, j):
    '''
    根据位置关系进行分类
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    '''
    ls_0 = [mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
                   ls_1 = [mat[i][j + 1], mat[i + 1][j], mat[i][j - 1]]
                                  ls_2 = [mat[i + 1][j], mat[i][j - 1], mat[i - 1][j]]
                                                 ls_3 = [mat[i][j - 1], mat[i - 1][j], mat[i][j+1]]
    '''
    a = ls.index(min(ls))
    if a == 0:
        ls_0 = [mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
        if ls_0[0] == min(ls_0):
            if ls_0[2] >= ls_0[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        elif ls_0[2] == min(ls_0):
            if ls_0[0] >= ls_0[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        else:
            if ls_0[2] >= ls_0[0]:   #ACBD
                return 2
            else:                    #ADBC
                return 3
    if a == 1:
        ls_1 = [mat[i][j + 1], mat[i + 1][j], mat[i][j - 1]]
        if ls_1[0] == min(ls_1):
            if ls_1[2] >= ls_1[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        elif ls_1[2] == min(ls_1):
            if ls_1[0] >= ls_1[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        else:
            if ls_1[2] >= ls_1[0]:   #ACBD
                return 2
            else:                    #ADBC
                return 3
    if a == 2:
        ls_2 = [mat[i + 1][j], mat[i][j - 1], mat[i - 1][j]]
        if ls_2[0] == min(ls_2):
            if ls_2[2] >= ls_2[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        elif ls_2[2] == min(ls_2):
            if ls_2[0] >= ls_2[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        else:
            if ls_2[2] >= ls_2[0]:   #ACBD
                return 2
            else:                    #ADBC
                return 3
    if a == 3:
        ls_3 = [mat[i][j - 1], mat[i - 1][j], mat[i][j+1]]
        if ls_3[0] == min(ls_3):
            if ls_3[2] >= ls_3[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        elif ls_3[2] == min(ls_3):
            if ls_3[0] >= ls_3[1]:   #ABCD
                return 0
            else:                    #ABDC
                return 1
        else:
            if ls_3[2] >= ls_3[0]:   #ACBD
                return 2
            else:                    #ADBC
                return 3

def point_num_type(mat,i, j):
    '''
    根据点的大小差值关系进行分类
    '''
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    diff = [ls[1]-ls[0], ls[2]-ls[1],ls[3]-ls[2]]
    if max(diff) == diff[0]:
        if ls[3]-ls[1] >= ls[1]-[0]:  # 0:4 ABCD
            return 0
        else:                         # 1:3 A/BCD
            if abs(ls[0]-mat[i,j]) <= abs(ls[3]-mat[i,j]):        #中心点像素更接近A
                return 1
            else:                                                 #中心点像素更接近D
                return 2
    elif max(diff) == diff[2]:
        if ls[2]-ls[0] >= ls[3]-ls[2]: # 0:4 ABCD
            return 0
        else:                          # 1:3 ABC/D
            if abs(ls[0] - mat[i, j]) <= abs(ls[3] - mat[i, j]):  #中心点像素更接近A
                return 3
            else:                                                 #中心点像素更接近D
                return 4
    else:                              # 2:2 AB/CD
        if abs(ls[0]-mat[i,j]) <= abs(ls[3]-mat[i,j]):            #中心点像素更接近A
            return 5
        else:                                                     #中心点像素更接近D
            return 6

def point_num_type1(mat,i, j):
    '''
    根据点的大小差值关系进行分类,只分4类
    ABCD A/BCD ABC/D AB/CD
    '''
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    diff = [ls[1]-ls[0], ls[2]-ls[1],ls[3]-ls[2]]
    if max(diff) == diff[0]:
        if ls[3]-ls[1] >= ls[1]-[0]:  # 0:4 ABCD
            return 0
        else:                         # 1:3 A/BCD
            return 1
    elif max(diff) == diff[2]:
        if ls[2]-ls[0] >= ls[3]-ls[2]: # 0:4 ABCD
            return 0
        else:                          # 1:3 ABC/D
            return 2
    else:                              # 2:2 AB/CD
            return 3

def sp_noise(image, prob):
    '''
    添加椒盐噪声
    :param image: 灰度图
    :param prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r1 = random.random()
            r2 = random.random()
            if r1 < prob:
                if r2 >= 0.5:
                    output[i][j] = 0
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def noise_point_count(mat,i,j):
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    while 0 in ls:
        ls.remove(0)
    while 255 in ls:
        ls.remove(255)
    return len(ls)

def compare_value(mat, list):
    x = neighbor4_4coef(mat,1,1,list)
    diff = abs(x-mat[1][1])
    return diff

def near_AD(mat):
    ls = [mat[1][0], mat[0][1], mat[1][2], mat[2][1]]
    if abs(mat[1][1]-min(ls)) <= abs((mat[1][1])-max(ls)):
        return 1
    else:
        return 0

def neighbor_8_mean(image, i, j):
    x = [
        image[i - 1][j - 1], image[i - 1][j], image[i - 1][j + 1],
        image[i][j - 1],     image[i][j + 1],
        image[i + 1][j - 1], image[i + 1][j], image[i + 1][j + 1]
        ]
    while 0 in x:
        x.remove(0)
    while 255 in x:
        x.remove(255)
    if x:
        return np.mean(x)
    else:
        return image[i][j]

def denoise_mat1(mat,i,j):
    '''
    用来处理4邻域点中存在1个噪声点的情况,返回1个替换过噪声点的3*3矩阵
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ret_mat = np.zeros((3, 3), dtype=np.float)
    ret_mat[1][0] = mat[i][j-1]
    ret_mat[0][1] = mat[i-1][j]
    ret_mat[1][1] = mat[i][j]
    ret_mat[1][2] = mat[i][j+1]
    ret_mat[2][1] = mat[i+1][j]
    if ls[0] == 0 or ls[0] == 255:  #左方点为噪声点
        if mat[i][j-2] != 0 and mat[i][j-2] != 255:
            ret_mat[1][0] = mat[i][j-2]
        elif mat[i - 1][j - 2] != 0 and mat[i - 1][j - 2] != 255:
            ret_mat[1][0] = mat[i - 1][j - 2]
        elif mat[i + 1][j - 2] != 0 and mat[i + 1][j - 2] != 255:
            ret_mat[1][0] = mat[i + 1][j - 2]
        elif mat[i][j-3] != 0 and mat[i][j-3] != 255:
            ret_mat[1][0] = mat[i][j - 3]
        elif mat[i - 1][j - 3] != 0 and mat[i - 1][j - 3] != 255:
            ret_mat[1][0] = mat[i - 1][j - 3]
        elif mat[i + 1][j - 3] != 0 and mat[i + 1][j - 3] != 255:
            ret_mat[1][0] = mat[i + 1][j - 3]
    elif ls[1] == 0 or ls[1] == 255:  #上方点为噪声点
        if mat[i-2][j] != 0 and mat[i-2][j] != 255:
            ret_mat[0][1] = mat[i-2][j]
        elif mat[i - 2][j - 1] != 0 and mat[i - 2][j - 1] != 255:
            ret_mat[0][1] = mat[i - 2][j - 1]
        elif mat[i - 2][j + 1] != 0 and mat[i - 2][j + 1] != 255:
            ret_mat[0][1] = mat[i - 2][j + 1]
        elif mat[i-3][j] != 0 and mat[i-3][j] != 255:
            ret_mat[0][1] = mat[i-3][j]
        elif mat[i - 3][j - 1] != 0 and mat[i - 3][j - 1] != 255:
            ret_mat[0][1] = mat[i - 3][j - 1]
        elif mat[i - 3][j + 1] != 0 and mat[i - 3][j + 1] != 255:
            ret_mat[0][1] = mat[i - 3][j + 1]
    elif ls[2] == 0 or ls[2] == 255: #右方点为噪声点
        if mat[i][j+2] != 0 and mat[i][j+2] != 255:
            ret_mat[1][2] = mat[i][j+2]
        elif mat[i - 1][j + 2] != 0 and mat[i - 1][j + 2] != 255:
            ret_mat[1][2] = mat[i - 1][j + 2]
        elif mat[i + 1][j + 2] != 0 and mat[i + 1][j + 2] != 255:
            ret_mat[1][2] = mat[i + 1][j + 2]
        elif mat[i][j+3] != 0 and mat[i][j-3] != 255:
            ret_mat[1][2] = mat[i][j + 3]
        elif mat[i - 1][j + 3] != 0 and mat[i - 1][j + 3] != 255:
            ret_mat[1][2] = mat[i - 1][j + 3]
        elif mat[i + 1][j + 3] != 0 and mat[i + 1][j + 3] != 255:
            ret_mat[1][2] = mat[i + 1][j + 3]
    elif ls[3] == 0 or ls[3] == 255:  #下方点为噪声点
        if mat[i+2][j] != 0 and mat[i+2][j] != 255:
            ret_mat[2][1] = mat[i+2][j]
        elif mat[i + 2][j - 1] != 0 and mat[i + 2][j - 1] != 255:
            ret_mat[2][1] = mat[i + 2][j - 1]
        elif mat[i + 2][j + 1] != 0 and mat[i + 2][j + 1] != 255:
            ret_mat[2][1] = mat[i + 2][j + 1]
        elif mat[i+3][j] != 0 and mat[i+3][j] != 255:
            ret_mat[2][1] = mat[i+3][j]
        elif mat[i + 3][j - 1] != 0 and mat[i + 3][j - 1] != 255:
            ret_mat[2][1] = mat[i - 3][j - 1]
        elif mat[i + 3][j + 1] != 0 and mat[i + 3][j + 1] != 255:
            ret_mat[2][1] = mat[i - 3][j + 1]

    return ret_mat

def denoise_mat2(mat,i,j):
    '''
    用来处理4邻域点中存在2个噪声点的情况,返回1个替换过噪声点的3*3矩阵
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ret_mat = np.zeros((3, 3), dtype=np.float)
    ret_mat[1][0] = mat[i][j-1]
    ret_mat[0][1] = mat[i-1][j]
    ret_mat[1][1] = mat[i][j]
    ret_mat[1][2] = mat[i][j+1]
    ret_mat[2][1] = mat[i+1][j]
    if ls[0] == 0 or ls[0] == 255:  #左方点为噪声点
        if mat[i][j-2] != 0 and mat[i][j-2] != 255:
            ret_mat[1][0] = mat[i][j-2]
        elif mat[i - 1][j - 2] != 0 and mat[i - 1][j - 2] != 255:
            ret_mat[1][0] = mat[i - 1][j - 2]
        elif mat[i + 1][j - 2] != 0 and mat[i + 1][j - 2] != 255:
            ret_mat[1][0] = mat[i + 1][j - 2]
        elif mat[i][j-3] != 0 and mat[i][j-3] != 255:
            ret_mat[1][0] = mat[i][j - 3]
        elif mat[i - 1][j - 3] != 0 and mat[i - 1][j - 3] != 255:
            ret_mat[1][0] = mat[i - 1][j - 3]
        elif mat[i + 1][j - 3] != 0 and mat[i + 1][j - 3] != 255:
            ret_mat[1][0] = mat[i + 1][j - 3]
    if ls[1] == 0 or ls[1] == 255:  #上方点为噪声点
        if mat[i-2][j] != 0 and mat[i-2][j] != 255:
            ret_mat[0][1] = mat[i-2][j]
        elif mat[i - 2][j - 1] != 0 and mat[i - 2][j - 1] != 255:
            ret_mat[0][1] = mat[i - 2][j - 1]
        elif mat[i - 2][j + 1] != 0 and mat[i - 2][j + 1] != 255:
            ret_mat[0][1] = mat[i - 2][j + 1]
        elif mat[i-3][j] != 0 and mat[i-3][j] != 255:
            ret_mat[0][1] = mat[i-3][j]
        elif mat[i - 3][j - 1] != 0 and mat[i - 3][j - 1] != 255:
            ret_mat[0][1] = mat[i - 3][j - 1]
        elif mat[i - 3][j + 1] != 0 and mat[i - 3][j + 1] != 255:
            ret_mat[0][1] = mat[i - 3][j + 1]
    if ls[2] == 0 or ls[2] == 255: #右方点为噪声点
        if mat[i][j+2] != 0 and mat[i][j+2] != 255:
            ret_mat[1][2] = mat[i][j+2]
        elif mat[i - 1][j + 2] != 0 and mat[i - 1][j + 2] != 255:
            ret_mat[1][2] = mat[i - 1][j + 2]
        elif mat[i + 1][j + 2] != 0 and mat[i + 1][j + 2] != 255:
            ret_mat[1][2] = mat[i + 1][j + 2]
        elif mat[i][j+3] != 0 and mat[i][j-3] != 255:
            ret_mat[1][2] = mat[i][j + 3]
        elif mat[i - 1][j + 3] != 0 and mat[i - 1][j + 3] != 255:
            ret_mat[1][2] = mat[i - 1][j + 3]
        elif mat[i + 1][j + 3] != 0 and mat[i + 1][j + 3] != 255:
            ret_mat[1][2] = mat[i + 1][j + 3]
    if ls[3] == 0 or ls[3] == 255:  #下方点为噪声点
        if mat[i+2][j] != 0 and mat[i+2][j] != 255:
            ret_mat[2][1] = mat[i+2][j]
        elif mat[i + 2][j - 1] != 0 and mat[i + 2][j - 1] != 255:
            ret_mat[2][1] = mat[i + 2][j - 1]
        elif mat[i + 2][j + 1] != 0 and mat[i + 2][j + 1] != 255:
            ret_mat[2][1] = mat[i + 2][j + 1]
        elif mat[i+3][j] != 0 and mat[i+3][j] != 255:
            ret_mat[2][1] = mat[i+3][j]
        elif mat[i + 3][j - 1] != 0 and mat[i + 3][j - 1] != 255:
            ret_mat[2][1] = mat[i - 3][j - 1]
        elif mat[i + 3][j + 1] != 0 and mat[i + 3][j + 1] != 255:
            ret_mat[2][1] = mat[i - 3][j + 1]

    return ret_mat

def denoise_mat3(mat,i,j):
    '''
    用来处理4邻域点中存在1个噪声点的情况,返回1个替换过噪声点的3*3矩阵
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ret_mat = np.zeros((3, 3), dtype=np.float)
    ret_mat[1][0] = mat[i][j-1]
    ret_mat[0][1] = mat[i-1][j]
    ret_mat[1][1] = mat[i][j]
    ret_mat[1][2] = mat[i][j+1]
    ret_mat[2][1] = mat[i+1][j]
    if ls[0] == 0 or ls[0] == 255:  #左方点为噪声点
        ls1 = [mat[i-1][j-1],mat[i+1][j-1]]
        while 0 in ls1:
            ls1.remove(0)
        while 255 in ls1:
            ls1.remove(255)
        if ls1:
            ret_mat[1][0] = np.median(ls1)
            return ret_mat
        ls2 = [mat[i-1][j-2], mat[i][j-2], mat[i+1][j-2]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[1][0] = np.median(ls2)
            return ret_mat
    if ls[1] == 0 or ls[1] == 255:  #上方点为噪声点
        ls1 = [mat[i-1][j-1],mat[i-1][j+1]]
        while 0 in ls1:
            ls1.remove(0)
        while 255 in ls1:
            ls1.remove(255)
        if ls1:
            ret_mat[0][1] = np.median(ls1)
            return ret_mat
        ls2 = [mat[i-2][j-1], mat[i-2][j], mat[i-2][j+1]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[0][1] = np.median(ls2)
            return ret_mat
    if ls[2] == 0 or ls[2] == 255: #右方点为噪声点
        ls1 = [mat[i-1][j+1],mat[i+1][j+1]]
        while 0 in ls1:
            ls1.remove(0)
        while 255 in ls1:
            ls1.remove(255)
        if ls1:
            ret_mat[1][2] = np.median(ls1)
            return ret_mat
        ls2 = [mat[i-1][j+2], mat[i][j+2], mat[i+1][j+2]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[1][2] = np.median(ls2)
            return ret_mat
    if ls[3] == 0 or ls[3] == 255:  #下方点为噪声点
        ls1 = [mat[i+1][j-1],mat[i+1][j+1]]
        while 0 in ls1:
            ls1.remove(0)
        while 255 in ls1:
            ls1.remove(255)
        if ls1:
            ret_mat[2][1] = np.median(ls1)
            return ret_mat
        ls2 = [mat[i+2][j-1], mat[i+2][j], mat[i+2][j+1]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[2][1] = np.median(ls2)
            return ret_mat

def denoise_mat4(mat,i,j):
    '''
    用来处理4邻域点中存在1个噪声点的情况,返回1个替换过噪声点的3*3矩阵
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ret_mat = np.zeros((3, 3), dtype=np.float)
    ret_mat[1][0] = mat[i][j-1]
    ret_mat[0][1] = mat[i-1][j]
    ret_mat[1][1] = mat[i][j]
    ret_mat[1][2] = mat[i][j+1]
    ret_mat[2][1] = mat[i+1][j]
    if ls[0] == 0 or ls[0] == 255:  #左方点为噪声点
        ls2 = [mat[i-1][j-2], mat[i][j-2], mat[i+1][j-2]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[1][0] = np.median(ls2)
            return ret_mat
    if ls[1] == 0 or ls[1] == 255:  #上方点为噪声点
        ls2 = [mat[i-2][j-1], mat[i-2][j], mat[i-2][j+1]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[0][1] = np.median(ls2)
            return ret_mat
    if ls[2] == 0 or ls[2] == 255: #右方点为噪声点
        ls2 = [mat[i-1][j+2], mat[i][j+2], mat[i+1][j+2]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[1][2] = np.median(ls2)
            return ret_mat
    if ls[3] == 0 or ls[3] == 255:  #下方点为噪声点
        ls2 = [mat[i+2][j-1], mat[i+2][j], mat[i+2][j+1]]
        while 0 in ls2:
            ls2.remove(0)
        while 255 in ls2:
            ls2.remove(255)
        if ls2:
            ret_mat[2][1] = np.median(ls2)
            return ret_mat
    return ret_mat

def medianfilter(mat,i,j):
    x = [
        mat[i - 1][j - 1], mat[i - 1][j], mat[i - 1][j + 1],
        mat[i][j - 1],                    mat[i][j + 1],
        mat[i + 1][j - 1], mat[i + 1][j], mat[i + 1][j + 1]
        ]
    while 0 in x:
        x.remove(0)
    while 255 in x:
        x.remove(255)
    x.sort()
    return np.median(x)

def meanfilter(mat, i, j):
    x = [
        mat[i - 1][j - 1], mat[i - 1][j], mat[i - 1][j + 1],
        mat[i][j - 1],     mat[i][j + 1],
        mat[i + 1][j - 1], mat[i + 1][j], mat[i + 1][j + 1]
        ]
    while 0 in x:
        x.remove(0)
    while 255 in x:
        x.remove(255)
    return np.mean(x)


def two_class(mat,i,j):
    ls4 = [mat[i][j-1],mat[i-1][j], mat[i][j+1], mat[i+1][j]]  #左上右下
    ls4_copy =ls4.copy()
    ls4.sort()
    if(ls4[3] - ls4[2]) > (ls4[1] - ls4[0]):  #4点中最大值属于1
        index = ls4_copy.index(max(ls4_copy)); #判断属于上下左右哪一个点
        if index == 0: #左
            ls_1 = [mat[i - 1][j - 1], mat[i - 1][j], mat[i - 2][j - 1], mat[i - 2][j]] #新建一个列表来判断
            ls_2 = [mat[i + 1][j - 1], mat[i + 1][j], mat[i + 2][j - 1], mat[i + 2][j]]
            if mat[i - 1][j - 1] == max(ls_1): #1：3
                if mat[i - 1][j - 1] - max(ls_1[1:]) > max(ls_1[1:]) - max(ls_1[1:]):
                    return 1
            if mat[i + 1][j - 1] == max(ls_2): #1：3
                if mat[i + 1][j - 1] - max(ls_2[1:]) > max(ls_2[1:]) - max(ls_2[1:]):
                    return 1
        elif index == 1: #上
            ls_1 = [mat[i - 1][j - 1], mat[i - 1][j - 2], mat[i - 2][j - 1], mat[i - 2][j - 2]] #新建一个列表来判断
            ls_2 = [mat[i - 1][j + 1], mat[i - 1][j + 2], mat[i - 2][j + 1], mat[i - 2][j + 2]]
            if mat[i - 1][j - 1] == max(ls_1): #1：3
                if mat[i - 1][j - 1] - max(ls_1[1:]) > max(ls_1[1:]) - max(ls_1[1:]):
                    return 1
            if mat[i - 1][j + 1] == max(ls_2): #1：3
                if mat[i - 1][j + 1] - max(ls_2[1:]) > max(ls_2[1:]) - max(ls_2[1:]):
                    return 1
        elif index == 2: #右
            ls_1 = [mat[i - 1][j + 1], mat[i - 1][j], mat[i - 2][j - 1], mat[i - 2][j]] #新建一个列表来判断
            ls_2 = [mat[i + 1][j - 1], mat[i + 1][j], mat[i + 2][j - 1], mat[i + 2][j]]
            if mat[i-1][j-1] == max(ls_1): #1：3
                if mat[i-1][j-1] - max(ls_1[1:]) > max(ls_1[1:]) - max(ls_1[1:]):
                    return 1
            if mat[i+1][j-1] == max(ls_2): #1：3
                if mat[i+1][j-1] - max(ls_2[1:]) > max(ls_2[1:]) - max(ls_2[1:]):
                    return 1
        else: #下
            ls_1 = [mat[i - 1][j - 1], mat[i - 1][j], mat[i - 2][j - 1], mat[i - 2][j]] #新建一个列表来判断
            ls_2 = [mat[i + 1][j - 1], mat[i + 1][j], mat[i + 2][j - 1], mat[i + 2][j]]
            if mat[i-1][j-1] == max(ls_1): #1：3
                if mat[i-1][j-1] - max(ls_1[1:]) > max(ls_1[1:]) - max(ls_1[1:]):
                    return 1
            if mat[i+1][j-1] == max(ls_2): #1：3
                if mat[i+1][j-1] - max(ls_2[1:]) > max(ls_2[1:]) - max(ls_2[1:]):
                    return 1


    else: #4点中最小值属于1
        index = ls4_copy.index(min(ls4_copy));  # 判断属于上下左右哪一个点
    return 0




mat = cv2.imread('C:/Users/AMDyes/Desktop/test_img/img256/house256.bmp', cv2.IMREAD_GRAYSCALE)

X = np.pad(mat, ((1,1), (1,1)), mode='symmetric')
X = X.astype(np.int64)
h, w = X.shape
e0 = e1 = e2 = e3 = np.float(0)
a0 = b0 = c0 = d0 = e0
a1 = b1 = c1 = d1 = e1
a2 = b2 = c2 = d2 = e2
a3 = b3 = c3 = d3 = e3
cnt = 0
res = np.empty([28, 4], dtype=float)

for k in range(4):
    for l in range(7):
        e0 = e1 = e2 = e3 = np.float(0)
        a0 = b0 = c0 = d0 = e0
        a1 = b1 = c1 = d1 = e1
        a2 = b2 = c2 = d2 = e2
        a3 = b3 = c3 = d3 = e3
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if point_loc_type(X,i,j) == k and point_num_type(X,i,j) == l:
                    cnt += 1
                    ls = [X[i][j - 1], X[i - 1][j], X[i][j + 1], X[i + 1][j]]
                    ls.sort()
                    a0 += ls[0] * ls[0]
                    b0 += ls[1] * ls[0]
                    c0 += ls[2] * ls[0]
                    d0 += ls[3] * ls[0]
                    e0 += X[i][j] * ls[0]

                    a1 += ls[0] * ls[1]
                    b1 += ls[1] * ls[1]
                    c1 += ls[2] * ls[1]
                    d1 += ls[3] * ls[1]
                    e1 += X[i][j] * ls[1]

                    a2 += ls[0] * ls[2]
                    b2 += ls[1] * ls[2]
                    c2 += ls[2] * ls[2]
                    d2 += ls[3] * ls[2]
                    e2 += X[i][j] * ls[2]

                    a3 += ls[0] * ls[3]
                    b3 += ls[1] * ls[3]
                    c3 += ls[2] * ls[3]
                    d3 += ls[3] * ls[3]
                    e3 += X[i][j] * ls[3]

        x = np.asarray([[a0, b0, c0, d0], [a1, b1, c1, d1], [a2, b2, c2, d2], [a3, b3, c3, d3]])
        y = np.asarray([e0, e1, e2, e3])
        ans = solve(x, y)
        index = k*7 + l
        res[index] = ans


print(res)











