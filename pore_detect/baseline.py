"""
最开始的毛孔检测方法
"""
import cv2
import numpy as np
import os

def usm(img, w=0.3):  # w影响较大，0.3较为合适
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    a = 1 / (1 - w)
    b = -w / (1 - w)
    usm_out = cv2.addWeighted(img, a, blur_img, b, 0)
    return usm_out

def detect_baseline(mask):
    adp_threshold = [121, 5]
    img_copy = mask.copy()
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray = usm(gray, w=0.5)
    
    # mask_area = np.sum(gray > 1)

    # 直接二值化效果优化
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                      adp_threshold[0], adp_threshold[1])
    threshold[mask[:, :, 1] < 10] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    binary = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    binary_draw = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20 and area < 2000:
            # 计算焦点坐标
            ellipse = cv2.fitEllipse(cnt)
            short = int(ellipse[1][0])
            long = int(ellipse[1][1])
            if short / long > 0.2:
                center_x = int(ellipse[0][0])
                center_y = int(ellipse[0][1])
                radius = np.clip(int(0.6 * ellipse[1][0]), 2, 15)
                # 利用bianry记录毛孔半径
                binary[center_y, center_x] = radius + 100
                cv2.drawContours(binary_draw,cnt,-1,255,-1)
                cv2.drawContours(img_copy, [cnt], 0, (0, 255, 255), -1)

    return binary_draw, img_copy

if __name__ == '__main__':
    path = r"F:\fh\center_white\0412\others_selected\cut\cut/"
    output_path = r"F:\fh\center_white\0412\others_selected\cut\baseline_output/"
    file_list = os.listdir(path)
    for file in file_list:
        name = file.split(".")[0]
        img = cv2.imread(path + file)
        result, real_result = detect_baseline(img)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("result", 500, 500)
        # cv2.imshow("result", real_result)
        # cv2.waitKey(0)
        cv2.imwrite(output_path + name + "_real_result.jpg", real_result)