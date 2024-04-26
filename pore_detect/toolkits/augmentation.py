"""
对图像做数据增强：gamma校正，hsv增强
input：一张图像
output：增强后的图像
"""

import cv2
import numpy as np
import random
import os
import time
class Augmentation:
    def __init__(self, h_gain=0.0, s_gain=0.5, v_gain=0.5, p=0.5):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        self.p = p

    def __call__(self, img, **kwargs):
        img = self.gamma_trans(img, float(1/2.2))
        if random.random() < self.p:
            self.augment_hsv(img, self.h_gain, self.s_gain, self.v_gain)
        return img

    def gamma_trans(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)
    
    def augment_hsv(self, im, hgain=0.0, sgain=0.5, vgain=0.5):
        hgain, sgain, vgain = self.h_gain, self.s_gain, self.v_gain
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            # r = np.array([hgain + 1, sgain + 1, vgain + 1])
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        
if __name__ == "__main__":
    input_folder = r"F:\fh\center_white\center_white"
    output_folder = r"F:\fh\center_white\s0.2"
    os.makedirs(output_folder, exist_ok=True)

    file_list = os.listdir(input_folder)
    for file in file_list:
        image = cv2.imread(os.path.join(input_folder, file))
        augmentation = Augmentation(p=1.0, s_gain=0.5, v_gain=0.2)
        augmented = augmentation(image)

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Gamma Corrected", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", 600, 1000)
        cv2.resizeWindow("Gamma Corrected", 600, 1000)
        cv2.imshow("Original", image)
        cv2.imshow("Gamma Corrected", augmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite(output_folder + file.split(".")[0] + "_augmented.png", augmented,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
