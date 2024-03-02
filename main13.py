# -*- coding: UTF-8 -*-
import  cv2
import numpy as np
# import skimage
import random
from coef_list import list
import matplotlib.pyplot as plt
# import operator
from skimage.metrics import structural_similarity as compare_ssim       #新增
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def neighbor4_4coef(mat, i, j, list):
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    x = ls[0] * list[0] + ls[1] * list[1] + ls[2] * list[2] + ls[3] * list[3]
    return round(x,8)

def point_loc_type(mat, i, j):
    '''
    根据位置关系进行分类
    '''
    ls = [mat[i][j - 1], mat[i - 1][j], mat[i][j + 1], mat[i + 1][j]]
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
    根据点的大小差值关系进行分类,判断中心点区域区域
    '''
    ls = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
    ls.sort()
    diff = [ls[1]-ls[0], ls[2]-ls[1],ls[3]-ls[2]]
    if max(diff) == diff[0]:
        if ls[3]-ls[1] >= ls[1]-ls[0]:  # 0:4 ABCD
            return 0
        else:                         # 1:3 A/BCD
            if abs(ls[0]-mat[i,j]) <= abs(ls[1]-mat[i,j]):        #中心点像素更接近A
                return 1
            else:                                                 #中心点像素更接近B
                return 2
    elif max(diff) == diff[2]:
        if ls[2]-ls[0] >= ls[3]-ls[2]: # 0:4 ABCD
            return 0
        else:                          # 1:3 ABC/D
            if abs(ls[2] - mat[i, j]) <= abs(ls[3] - mat[i, j]):  # 中心点像素更接近A
                return 3
            else:                                                 #中心点像素更接近B
                return 4
    else:                              # 2:2 AB/CD
        if abs(ls[1]-mat[i,j]) <= abs(ls[2]-mat[i,j]):            #中心点像素更接近A
            return 5
        else:                                                     #中心点像素更接近B
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
        else:                          # 3:1 ABC/D
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
    ret_mat = np.zeros((3, 3), dtype=np.float64)
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

def lower_noise_density(mat, k, h, w):
    mat1 = mat.copy()
    for i in range(2, h - 2 - k + 1, 1):
        for j in range(2, w - 2 - k + 1, 1):
            ls = []
            for mi in range(k):
                for mj in range(k):
                    ls.append(mat1[i + mi][j + mj])

            while 0 in ls:
                ls.remove(0)
            while 255 in ls:
                ls.remove(255)
            if len(ls) == 1:
                for ni in range(k):
                    for nj in range(k):
                        mat1[i + ni][j + nj] = ls[0]
    return mat1

def get_noise_degree(mat, h, w):
    noise_count = 0
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if mat[i][j] == 0 or mat[i][j] == 255:
                noise_count += 1
    nd = noise_count / (h - 2) / (w - 2)
    return nd

def get_noise_cnt(mat, h, w):
    noise_count = 0
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if mat[i][j] == 0 or mat[i][j] == 255:
                noise_count += 1
    return noise_count

img = cv2.imread('src/Lena.bmp', cv2.IMREAD_GRAYSCALE)  #读取灰度图
noise_img = sp_noise(img, 0.8) #添加噪声
padded_img = np.pad(img, ((1, 1), (1, 1)), mode='symmetric')
h, w = noise_img.shape
sub_h, sub_w = h // 2, w // 2 # 1/4子图的高宽
sub_image1 = np.zeros((sub_h, sub_w))
sub_image2 = np.zeros((sub_h, sub_w))
sub_image3 = np.zeros((sub_h, sub_w))
sub_image4 = np.zeros((sub_h, sub_w))
for i in range(0, sub_h): # 分割为4张子图
    for j in range(0, sub_w):
        sub_image1[i][j] = img[i * 2][j * 2]
        sub_image2[i][j] = img[i * 2][j * 2 + 1]
        sub_image3[i][j] = img[i * 2 + 1][j * 2]
        sub_image4[i][j] = img[i * 2 + 1][j * 2 + 1]

padded_noise_mat = np.pad(noise_img, ((2,2), (2,2)), mode='symmetric') #边缘填充
padded_noise_mat = padded_noise_mat.astype(np.float64)
h, w = padded_noise_mat.shape

padded_sub_image1 = np.pad(sub_image1, ((2,2), (2,2)), mode='symmetric') #子图边缘填充
padded_sub_image2 = np.pad(sub_image2, ((2,2), (2,2)), mode='symmetric')
padded_sub_image3 = np.pad(sub_image3, ((2,2), (2,2)), mode='symmetric')
padded_sub_image4 = np.pad(sub_image4, ((2,2), (2,2)), mode='symmetric')
padded_sub_images = [padded_sub_image1, padded_sub_image2, padded_sub_image3, padded_sub_image4]

t = [3/4, 8/9, 15/16] #噪声密度阈值
# 获取噪声密度
nd = get_noise_degree(padded_noise_mat, h, w)
print(nd)

# 对高密度噪声降噪
while(nd >= 0):
    padded_noise_mat = lower_noise_density(padded_noise_mat, 4, h, w)
    nd0 = get_noise_degree(padded_noise_mat, h, w)
    print('4: ', nd0)
    if(nd0 < nd):
        nd = nd0
    else:
        break
while(nd >= 0):
    padded_noise_mat = lower_noise_density(padded_noise_mat, 3, h, w)
    nd0 = get_noise_degree(padded_noise_mat, h, w)
    print('3: ', nd0)
    if(nd0 < nd):
        nd = nd0
    else:
        break
while(nd >= 0):
    padded_noise_mat = lower_noise_density(padded_noise_mat, 2, h, w)
    nd0 = get_noise_degree(padded_noise_mat, h, w)
    print('2: ', nd0)
    if(nd0 < nd):
        nd = nd0
    else:
        break

denoise_img = padded_noise_mat.copy()

noise_cnt = get_noise_cnt(denoise_img, h, w)
roll_flag = 1

while(1):
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            if padded_noise_mat[i][j] == 0 or padded_noise_mat[i][j] == 255:
                #xx[i][j] = neighbor_8_mean(padded_noise_mat,i,j)
                #xx[i][j] = medianfilter(padded_noise_mat,i,j)
                if noise_point_count(padded_noise_mat,i,j) == 4 and roll_flag == 1:       #4邻域点均为信号点
                    loc = point_loc_type(padded_noise_mat,i,j)
                    num = point_num_type1(padded_noise_mat,i,j)
                    index_d = (loc * 7 + num * 2)
                    index_a = (loc * 7 + num * 2) - 1
                    if num == 0:             #0:4不用判断和A、D哪个更接近
                        ls_d = list.dp[index_d]
                        temp_d = neighbor4_4coef(padded_noise_mat, i, j, ls_d)  # 临时估计值
                        denoise_img[i, j] = temp_d
                        continue
                    elif num == 1:             #1:3 A/BCD
                        ls_d = list.dp[index_d]  # 靠近D的一组系数
                        denoise_img[i, j] = neighbor4_4coef(padded_noise_mat, i, j, ls_d) #直接将中心点归为3个点的1类
                        continue
                    elif num == 2:             #1:3 ABC/D
                        ls_a = list.dp[index_a]  # 靠近A的一组系数
                        denoise_img[i, j] = neighbor4_4coef(padded_noise_mat, i, j, ls_a) #直接将中心点归为3个点的1类
                        continue
                    # xx[i, j] = np.mean([padded_noise_mat[i][j-1], padded_noise_mat[i-1][j], padded_noise_mat[i][j+1], padded_noise_mat[i+1][j]])
                    #2:2 AB/CD
                    else:
                        ls_a = list.dp[max(index_a, 0)]           #靠近A的一组系数
                        ls_d = list.dp[index_d]                  #靠近D的一组系数
                        temp_a = neighbor4_4coef(padded_noise_mat,i,j,ls_a)  #临时估计值
                        temp_d = neighbor4_4coef(padded_noise_mat,i,j,ls_d)  #临时估计值
                        ls = [padded_noise_mat[i][j-1], padded_noise_mat[i-1][j], padded_noise_mat[i][j+1], padded_noise_mat[i+1][j]]
                        ls.sort()
                        mean_a = (ls[0]+ls[1]) / 2
                        mean_d = (ls[2]+ls[3]) / 2
                        if abs(temp_a-mean_a) < abs(temp_d-mean_d):
                            denoise_img[i][j] = temp_a
                        else:
                            denoise_img[i][j] = temp_d
                        mat_a = [[0, denoise_img[i - 2, j], 0], [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]], [0, temp_a, 0]]  #中心点上方矩阵，上方点均为已处理点
                        mat_d = [[0, denoise_img[i - 2, j], 0], [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]], [0, temp_d, 0]]
                        loc_a = point_loc_type(mat_a,1,1)
                        num_a = point_num_type1(mat_a,1,1)
                        loc_d = point_loc_type(mat_d,1,1)
                        num_d = point_num_type1(mat_d,1,1)
                        index_a = (loc_a * 7 + num_a * 2)
                        index_d = (loc_d * 7 + num_d * 2)
                        if num_a != 0 and near_AD(mat_a) == 1:  #num_a==0说明是0：4，不需要判断和A、D哪个更接近
                            index_a -= 1                        #更接近A, 则要-1，因为列表是按靠近A 靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                        ls_a = list.dp[max(index_a, 0)]           #靠近A的一组系数
                        ls_d = list.dp[max(index_d, 0)]           #靠近D的一组系数
                        if compare_value(mat_a,ls_a) < compare_value(mat_d, ls_d):   #选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                            denoise_img[i, j] = temp_a
                        else:
                            denoise_img[i, j] = temp_d
                elif noise_point_count(padded_noise_mat,i,j) == 3 and roll_flag == 2: #有1个噪声点，在子图中对该噪声点处理
                    sub_i, sub_j = i // 2, j // 2    #子图坐标
                    if i % 2 == 0 and j % 2 == 0:    #子图1
                        temp_mat = denoise_mat1(padded_sub_image1, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    elif i % 2 == 0 and j % 2 == 1:  #子图2
                        temp_mat = denoise_mat1(padded_sub_image2, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    elif i % 2 == 1 and j % 2 == 0:  #子图3
                        temp_mat = denoise_mat1(padded_sub_image3, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    else: #子图4 i % 2 == 1 and j % 2 == 1
                        temp_mat = denoise_mat1(padded_sub_image4, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d

                elif (noise_point_count(padded_noise_mat, i, j) == 2 or noise_point_count(padded_noise_mat, i, j) == 1 or noise_point_count(padded_noise_mat, i, j) == 0) and roll_flag == 3:   #有2个噪声点
                    sub_i, sub_j = i // 2, j // 2    #子图坐标
                    if i % 2 == 0 and j % 2 == 0:    #子图1
                        temp_mat = denoise_mat1(padded_sub_image1, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    elif i % 2 == 0 and j % 2 == 1:  #子图2
                        temp_mat = denoise_mat1(padded_sub_image2, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    elif i % 2 == 1 and j % 2 == 0:  #子图3
                        temp_mat = denoise_mat1(padded_sub_image3, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                    else: #子图4 i % 2 == 1 and j % 2 == 1
                        temp_mat = denoise_mat1(padded_sub_image4, sub_i, sub_j)
                        loc = point_loc_type(temp_mat, 1, 1)
                        num = point_num_type1(temp_mat, 1, 1)
                        index_d = (loc * 7 + num * 2)
                        index_a = (loc * 7 + num * 2) - 1
                        if num == 0:  # 0:4不用判断和A、D哪个更接近
                            ls_d = list.dp[index_d]
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            continue
                        elif num == 1:  # 1:3 A/BCD
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 直接将中心点归为3个点的1类
                            continue
                        elif num == 2:  # 1:3 ABC/D
                            ls_a = list.dp[index_a]  # 靠近A的一组系数
                            denoise_img[i, j] = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 直接将中心点归为3个点的1类
                            continue
                        else:           # 2:2 AB/CD
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[index_d]  # 靠近D的一组系数
                            temp_a = neighbor4_4coef(temp_mat, 1, 1, ls_a)  # 临时估计值
                            temp_d = neighbor4_4coef(temp_mat, 1, 1, ls_d)  # 临时估计值
                            mat_a = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_a, 0]]  # 中心点上方矩阵，上方点均为已处理点
                            mat_d = [[0, denoise_img[i - 2, j], 0],
                                     [denoise_img[i - 1, j - 1], denoise_img[i - 1, j], denoise_img[i - 1, j + 1]],
                                     [0, temp_d, 0]]
                            loc_a = point_loc_type(mat_a, 1, 1)
                            num_a = point_num_type1(mat_a, 1, 1)
                            loc_d = point_loc_type(mat_d, 1, 1)
                            num_d = point_num_type1(mat_d, 1, 1)
                            index_a = (loc_a * 7 + num_a * 2)
                            index_d = (loc_d * 7 + num_d * 2)
                            if num_a != 0 and near_AD(mat_a) == 1:  # num_a==0说明是0：4，不需要判断和A、D哪个更接近
                                index_a -= 1  # 更接近A, 则要-1，因为列表是按靠近A再靠近D的顺序排布的
                            if num_d != 0 and near_AD(mat_d) == 1:
                                index_d -= 1
                            ls_a = list.dp[max(index_a, 0)]  # 靠近A的一组系数
                            ls_d = list.dp[max(index_d, 0)]  # 靠近D的一组系数
                            if compare_value(mat_a, ls_a) < compare_value(mat_d,
                                                                          ls_d):  # 选择以该估计值作为4邻域点，对所在中心点估计值与中心值较接近的估计值
                                denoise_img[i, j] = temp_a
                            else:
                                denoise_img[i, j] = temp_d
                else:
                    continue
    noise_cnt0 = get_noise_cnt(denoise_img, h, w)
    print(roll_flag, noise_cnt0)
    if(noise_cnt0 < noise_cnt):
        roll_flag = 1
        padded_noise_mat = denoise_img
        noise_cnt = noise_cnt0
    else:
        if(roll_flag < 3):
            roll_flag += 1
        else:
            break

#将去噪图转化成与原图相同尺寸
X = np.zeros(img.shape)
for i in range(2, h - 2):
    for j in range(2, w - 2):
        X[i - 2][j - 2] = denoise_img[i][j]

X = X.astype(np.int64)
XX = X

psnr1 = compare_psnr(img, XX, data_range = 255)
ssim1 = compare_ssim(img, XX, data_range = 255)
print('psnr =',psnr1,'ssim =',ssim1)


plt.imshow(XX,cmap = plt.get_cmap('gray')) # 显示图片
plt.axis('off')                            # 不显示坐标轴
plt.savefig('src/test.png')
