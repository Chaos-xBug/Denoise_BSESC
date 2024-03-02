# -*- coding: UTF-8 -*-
import  cv2
import numpy as np
from scipy.linalg import solve
import skimage
import matplotlib.pyplot as plt

def neighbor4_4coef(mat, i, j, list):
    ls = [mat[i-1][j-1], mat[i-1][j+1], mat[i+1][j-1], mat[i+1][j+1]]
    ls.sort()
    return ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2] + ls[3] * list[3]

def neighbor4_3coef(mat, i, j, list):
    ls = [mat[i-1][j-1], mat[i-1][j+1], mat[i+1][j+1]]
    ls.sort()
    return ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2]

def point_type1(mat, i, j):#4个点不可分
    ls = [mat[i-1,j-1], mat[i-1,j+1], mat[i+1,j-1], mat[i+1,j+1]]
    ls.sort();
    diff = [ls[1] - ls[0], ls[2] - ls[1], ls[3] - ls[2]]
    if max(diff) == diff[0]:
        if ls[3] - ls[1] >= ls[1] - [0]:
            return 0
        else:
            return 1
    elif max(diff) == diff[2]:
        if ls[2] - ls[0] >= ls[3] - ls[2]:
            return 0
        else:
            return 1

def point_type2(mat, i, j):
    '''
    ABC/D, 1:3
    17 --- 18
     |     |
    24 --- 20
    '''

    ls = [mat[i-1,j-1], mat[i-1,j+1], mat[i+1,j-1], mat[i+1,j+1]]
    ls_l = [mat[i-1,j-1], mat[i-1,j+1], mat[i+1,j+1]]
    ls_l.sort()
    if mat[i+1,j-1] == max(ls):
        if max(ls) - ls_l[2] > ls_l[2]-ls_l[0]:
            return 0;

def point_type3(mat, i, j):
    '''
    ABC/D, 1:3
    17 --- 18
     |     |
    20 --- 25
    '''

    ls = [mat[i-1,j-1], mat[i-1,j+1], mat[i+1,j-1], mat[i+1,j+1]]
    ls_l = [mat[i-1,j-1], mat[i-1,j+1], mat[i+1,j-1]]
    ls_l.sort()
    if mat[i+1,j+1] == max(ls):
        if max(ls) - ls_l[2] > ls_l[2]-ls_l[0]:
            return 0;

mat = cv2.imread('C:/Users/AMDyes/Desktop/result/lena512.bmp', cv2.IMREAD_GRAYSCALE)
h, w = mat.shape
h, w = h + 2, w + 2
X = np.zeros((h,w), dtype=np.int64)
pt = np.zeros(mat.shape, dtype=np.uint8)  #存放点的类型
X[0, 0] = mat[0, 0]
X[h - 1, 0] = mat[h - 3, 0]
X[0, w - 1] = mat[0, w - 3]
X[h - 1, w - 1] = mat[h - 3, w - 3]
#采用4个点进行拟合时，需要4个系数+1个常数项，以及4条方程

e0 = e1 = e2 = e3 = np.float(0)
a0 = b0 = c0 = d0 = e0
a1 = b1 = c1 = d1 = e1
a2 = b2 = c2 = d2 = e2
a3 = b3 = c3 = d3 = e3



XX = X
for i in range(1, h - 1):
    for j in range(1, w - 1):
        if point_type3(X,i,j) == 0:
            XX[i, j-1] = neighbor4_3coef(X, i, j, list10)

xxx = np.zeros(mat.shape, dtype=np.uint8)
for i in range(1, h - 1):
    for j in range(1, w - 1):
        xxx[i-1][j-1] = XX[i][j]
diff = cv2.absdiff(mat, xxx)
cv2.imshow("img", xxx)
cv2.waitKey(0)




