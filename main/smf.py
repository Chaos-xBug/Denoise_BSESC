# -*- coding:UTF-8 -*-
import  cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import skimage
import  xlsxwriter
import os

def smf(image,i,j):
    x = [
        image[i - 1][j - 1], image[i - 1][j], image[i - 1][j + 1],
        image[i][j - 1],                    image[i][j + 1],
        image[i + 1][j - 1], image[i + 1][j], image[i + 1][j + 1]
        ]
    y = image[i][j]
    while 0 in x:
        x.remove(0)
    while 255 in x:
        x.remove(255)
    if x:
        x.sort()
        return np.median(x)
    else:
        return y
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


def point_type(image, i, j):
    mat = np.asarray(image[i - 1][j - 1], image[i - 1][j], image[i - 1][j + 1], image[i][j - 1], image[i][j + 1], image[i][j + 1], image[i + 1][j - 1], image[i + 1][j], image[i + 1][j + 1]).reshape(3,3)


img = cv2.imread('C:/Users/AMDyes/Desktop/result/lena512.bmp', cv2.IMREAD_GRAYSCALE)  #读取灰度图
noise_img = cv2.imread('C:/Users/AMDyes/Desktop/noise_img/0.1noise.bmp', cv2.IMREAD_GRAYSCALE)  #读取灰度图
#noise_img = sp_noise(img, 0.1)
#directory = r'C:\Users\AMDyes\Desktop\noise_img'
'''

plt.imshow(noise_img, cmap = 'gray')       # 绘制图片
plt.axis('off')                            # 不显示坐标轴
plt.show()                                 # 显示图片


os.chdir(directory)
#filename = 'savedImage.bmp'
#cv2.imwrite(filename, noise_img)
ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for x in ls:
    noise_img = sp_noise(img, x)
    filename = str(x) + 'noise.bmp'
    cv2.imwrite(filename, noise_img)
'''
noise_img1 = np.pad(noise_img, ((1, 1), (1, 1)), mode='symmetric')
denoise_img1 = noise_img1.copy()

for i in range(1,noise_img1.shape[0] - 1):
    for j in range(1, noise_img1.shape[1] - 1):
        if noise_img1[i][j] == 0 or noise_img1[i][j] == 255:
            denoise_img1[i][j] = smf(noise_img1, i, j)

denoise_img = np.zeros(img.shape)
for i in range(1, denoise_img1.shape[0] - 1):
    for j in range(1, denoise_img1.shape[1] - 1):
        denoise_img[i - 1][j - 1] = denoise_img1[i][j]
psnr = skimage.measure.compare_psnr(img, denoise_img, data_range = 255)
ssim = skimage.measure.compare_ssim(img, denoise_img, data_range = 255)
print(psnr,ssim)
plt.imshow(denoise_img, cmap = 'gray')        # 绘制图片
plt.axis('off')                               # 不显示坐标轴
plt.show()                                    # 显示图片

workbook = xlsxwriter.Workbook('C:/Users/AMDyes/Desktop/result/lena512.xlsx') # 建立文件
worksheet1 = workbook.add_worksheet('original_img')
worksheet2 = workbook.add_worksheet('noise_img')
worksheet3 = workbook.add_worksheet('denoise_img')
worksheet4 = workbook.add_worksheet('diff')
cell_format1 = workbook.add_format()
cell_format1.set_pattern(1)
cell_format1.set_bg_color('red')





