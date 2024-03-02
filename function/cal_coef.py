# -*- coding: UTF-8 -*-
'''
计算4邻域有1个噪声点系数
'''
import cv2
import numpy as np
from scipy.linalg import solve
import math
def cal_psnr(image, denoise_image):
    '''
    计算PSNR
    :param image:原图
    :param denoise_image:去噪图
    :return: PSNR
    '''
    m = image.shape[0]
    n = image.shape[1]
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += (image[i][j] - denoise_image[i][j]) * (image[i][j] - denoise_image[i][j])
    MSE = sum / (m * n)
    psnr = 10 * math.log(255 * 255 / MSE, 10)
    return psnr

def neighbor4_4coef(mat, i, j, list):
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    return ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2] + ls[3] * list[3]

def neighbor4_3coef(mat, i, j, list, index):
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ls.sort()
    if index == 0:
        return ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2]
    else:
        return ls[1] * list[0] + ls[2] * list[1] + ls[3] * list[2]

def neighbor4_2coef(mat, i, j, list, index):
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    ls.sort()
    if index == 0:
        return ls[0] * list[0] + ls[1] * list[1]
    else:
        return ls[2] * list[0] + ls[3] * list[1]

def noise_point_count(mat,i,j):
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
    while 0 in ls:
        ls.remove(0)
    while 255 in ls:
        ls.remove(255)
    return len(ls)

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
def point_num_type2(mat,i, j):
    '''
    根据点的大小差值关系进行分类
    '''
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    diff = [ls[1]-ls[0], ls[2]-ls[1],ls[3]-ls[2]]
    if diff[0] <= 5:
        if diff[1] <= 5:
            if diff[2] <= 5:
                return 0
            else:
                return 1
        else:
            if diff[2] <= 5:
                return 2
            else:
                return 3
    else:
        if diff[1] <= 5:
            if diff[2] <= 5:
                return 4
            else:
                return 5
        else:
            if diff[2] <= 5:
                return 6
            else:
                return 7
list = []
mat = cv2.imread('C:/Users/AMDyes/Desktop/result/lena512.bmp', cv2.IMREAD_GRAYSCALE)  #读取灰度图
noise_mat = cv2.imread('C:/Users/AMDyes/Desktop/result/noise_img10%.bmp', cv2.IMREAD_GRAYSCALE)
mat = mat.astype(np.int64)
noise_mat = noise_mat.astype(np.int64)
padded_mat = np.pad(mat, ((2,2), (2,2)), mode='symmetric')
padded_noise_mat = np.pad(noise_mat, ((2,2), (2,2)), mode='symmetric')
xx = padded_noise_mat
h, w = padded_noise_mat.shape
e0 = e1 = e2 = e3 = np.float(0)
a0 = b0 = c0 = d0 = e0
a1 = b1 = c1 = d1 = e1
a2 = b2 = c2 = d2 = e2
a3 = b3 = c3 = d3 = e3
cnt = 0
xx = padded_mat.copy()
for t in range(1):
    e0 = e1 = e2 = e3 = np.float(0)
    a0 = b0 = c0 = d0 = e0
    a1 = b1 = c1 = d1 = e1
    a2 = b2 = c2 = d2 = e2
    a3 = b3 = c3 = d3 = e3
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if point_loc_type(padded_mat, i, j) == 1 and point_num_type(padded_mat, i, j) == 0:
                if point_num_type2(padded_mat, i, j) == 0:
                    ls = [padded_mat[i][j-1], padded_mat[i-1][j], padded_mat[i][j+1], padded_mat[i+1][j]]
                    ls.sort()
                    a0 += ls[0] * ls[0]
                    b0 += ls[1] * ls[0]
                    c0 += ls[2] * ls[0]
                    d0 += ls[3] * ls[0]
                    e0 += padded_mat[1][1] * ls[0]

                    a1 += ls[0] * ls[1]
                    b1 += ls[1] * ls[1]
                    c1 += ls[2] * ls[1]
                    d1 += ls[3] * ls[1]
                    e1 += padded_mat[1][1] * ls[1]

                    a2 += ls[0] * ls[2]
                    b2 += ls[1] * ls[2]
                    c2 += ls[2] * ls[2]
                    d2 += ls[3] * ls[2]
                    e2 += padded_mat[1][1] * ls[2]

                    a3 += ls[0] * ls[3]
                    b3 += ls[1] * ls[3]
                    c3 += ls[2] * ls[3]
                    d3 += ls[3] * ls[3]
                    e3 += padded_mat[1][1] * ls[3]

    x = np.array([[a0, b0, c0, d0], [a1, b1, c1, d1], [a2, b2, c2, d2], [a3, b3, c3, d3]])
    y = np.array([e0, e1, e2, e3])
    ans = solve(x, y)
    list.append(ans)

print(list)





'''
import  math
for i in range(len(ls)):
    ls1.append(10 * math.log(262144 // ls[i], 10))
    ls1[i] = round(ls1[i],2)
print(ls1)

                if noise_point_count(padded_noise_mat,i,j) == 2:       #4邻域点均为信号点
                    temp_mat = denoise_mat3(padded_noise_mat,i,j)
                    temp_mat[1][1] = padded_mat[i][j]
                    if point_loc_type(temp_mat, 1, 1) == k and point_num_type(temp_mat, 1, 1) == t:
                        ls = [temp_mat[1][0], temp_mat[0][1], temp_mat[1][2], temp_mat[2][1]]
                        ls.sort()
                        a0 += ls[0] * ls[0]
                        b0 += ls[1] * ls[0]
                        c0 += ls[2] * ls[0]
                        d0 += ls[3] * ls[0]
                        e0 += temp_mat[1][1] * ls[0]

                        a1 += ls[0] * ls[1]
                        b1 += ls[1] * ls[1]
                        c1 += ls[2] * ls[1]
                        d1 += ls[3] * ls[1]
                        e1 += temp_mat[1][1] * ls[1]

                        a2 += ls[0] * ls[2]
                        b2 += ls[1] * ls[2]
                        c2 += ls[2] * ls[2]
                        d2 += ls[3] * ls[2]
                        e2 += temp_mat[1][1] * ls[2]

                        a3 += ls[0] * ls[3]
                        b3 += ls[1] * ls[3]
                        c3 += ls[2] * ls[3]
                        d3 += ls[3] * ls[3]
                        e3 += temp_mat[1][1] * ls[3]

        x = np.array([[a0,b0,c0,d0],[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]])
        y = np.array([e0,e1,e2,e3])
        ans = solve(x,y)
        list.append(ans)
'''














