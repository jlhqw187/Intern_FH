"""
罐内气泡计数
input: 罐内图片
output: 标注罐内所有气泡的图片
"""
import cv2
import numpy as np
import os

def lep_preprocess(image, sigma_s=150, sigma_r=0.1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    
    ## 距离转换滤波器
    base_layer = cv2.ximgproc.dtFilter(image, image, sigma_s, sigma_r)
    detail_layer = image - base_layer
    base_layer = np.uint8(base_layer * 255)

    
    detail_layer = cv2.normalize(detail_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    detail_layer = np.uint8(detail_layer)
    

    return base_layer, detail_layer

def clahe_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_y = image[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image_y)
    return clahe_image

def usm(img, w=0.3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自动确定高斯核的大小，标准差为5，越大越模糊
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    a = 1 / (1 - w)
    b = -w / (1 - w)
    usm_out = cv2.addWeighted(img, a, blur_img, b, 0)
    return usm_out

input_path = r"F:\fh\bubble_count\input/"
output_path = r"F:\fh\bubble_count\contrast/"
file_list = os.listdir(input_path)

for file in file_list:
    img = cv2.imread(input_path + file)
    img = img[10:530, 150:800, :]
    img_copy, img_poly = img.copy(), img.copy()
    cv2.imwrite(output_path + file.split(".")[0] + ".jpg", img)

    gray = usm(img)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(output_path + file.split(".")[0] + "_thresh.jpg", thresh)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        cv2.drawContours(img_poly, [contour], -1, (0, 255, 0), -1)
        cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 1)


    cv2.imwrite(output_path + file.split(".")[0] + f"_detect_{len(contours)}.jpg", img_copy)
    cv2.imwrite(output_path + file.split(".")[0] + f"_fill_{len(contours)}.jpg", img_poly)

    # cv2.waitKey(0)