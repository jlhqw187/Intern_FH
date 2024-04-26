"""
从修改过的real_result上提取出修改过的label并生成新的real_result
input: ori_path(毛孔矩形图片文件夹), input_path(用画图修改过的real_result文件夹), 
       output_path(label生成的位置), new_real_result_path(新的real_real_result生成的位置)
output: output_path(label生成的位置), new_real_result_path(新的real_real_result生成的位置)
"""

import cv2
import numpy as np
import os

ori_path = r"F:\fh\center_white\0412\others_selected\0415ok\0415ok\cleaned_0418\cleaned_0418\washing\img/"
input_path = r"F:\fh\center_white\0412\others_selected\0415ok\0415ok\cleaned_0418\cleaned_0418\washing\real_img/"
output_path = r"F:\fh\center_white\0412\others_selected\0415ok\0415ok\cleaned_0418\cleaned_0418\washing\label/"
new_real_result_path = r"F:\fh\center_white\0412\others_selected\0415ok\0415ok\cleaned_0418\cleaned_0418\washing\real_result/"

file_list = os.listdir(input_path)

lower_bound = np.array([0, 250, 0])
upper_bound = np.array([0, 255, 0])

for file in file_list:
    name = file.split(".")[0]
    name = name.split("_real_result")[0]
    img = cv2.imread(input_path + file)
    ori_img =  cv2.imread(ori_path + name + ".png")
    mask = cv2.inRange(img, lower_bound, upper_bound)
    ori_img[mask == 255] = (0, 255, 0)
    print(name)
    

    cv2.imwrite(output_path + name + "_label.png", mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite(new_real_result_path + name + "_real_result.png", ori_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])