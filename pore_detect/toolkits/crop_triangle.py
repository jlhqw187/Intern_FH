"""
从只包含人脸部位的图片中获取矩形图片。
input: input_path包含人脸部位的图片文件夹
output: output_path获取到的矩形图片文件夹
"""
import numpy as np
import cv2
import os
import re

input_path = r"F:\fh\center_white\0412\others_selected\cut/"
output_path = r"F:\fh\center_white\0412\others_selected\triangle/"
file_list = os.listdir(input_path)
left_pattern = re.compile(r'.*left.*cut.*')
right_pattern = re.compile(r'.*right.*cut.*')
left_files = [file for file in file_list if left_pattern.match(file)]
right_files = [file for file in file_list if right_pattern.match(file)]
for file in left_files:
    img = cv2.imread(input_path + file)
    height, width, _ = img.shape
    bar_x = width // 3
    bar_y = height // 3

    # cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    img = img[bar_y:2*bar_y, bar_x:2*bar_x]
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    cv2.imwrite(output_path + file.split(".")[0] + "_crop.jpg", img)

for file in right_files:
    img = cv2.imread(input_path + file)
    height, width, _ = img.shape
    bar_x = width // 3
    bar_y = height // 3

    # cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    img = img[bar_y:2*bar_y, bar_x:2*bar_x]
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    cv2.imwrite(output_path + file.split(".")[0] + "_crop.jpg", img)
