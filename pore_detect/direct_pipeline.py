"""
对人脸图片按部位进行分割并进行毛孔检测
input: 人脸图片文件夹, 部位名称
output: 切割出的部位的毛孔检测结果文件夹
"""
import numpy as np
import cv2
import os
import cut
import pore_detect
import time
import re
import direct

input_path = r"F:\fh\center_white\0411\selected/input/"
output_path = r"F:\fh\center_white\0411\selected\output/"

parts_list = ['left_pore_area','right_pore_area']
for file in os.listdir(input_path):
        name = file.split(".")[0]
        path = os.path.join(input_path,file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = pore_detect.usm(img)
        img_cp, ori = img.copy(), img.copy()
        
        # keypoint是mp自带检测出的人脸关键点
        keypoint = cut.get_face_mesh(img_cp)


        if len(keypoint) ==0:
            continue
        

        cut_img = cut.CutImg(keypoint)
        a = time.time()

        # face_mask是onnx的output，为原始大小
        face_mask = cut.face_seg(img)

        # 获取通过mask分割出的人脸
        img[face_mask == 0] = [0, 0, 0]

        for part in parts_list:
            part_name = part
            part = [part]
            mask_new, xyxy, keypoint_parts = cut_img.get_parts(img, part)
            
            # 局部矩形图像rect_img
            rect_img = ori[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]

            # parts_mask是物体在局部图像上的掩膜
            parts_mask = np.zeros(mask_new.shape,dtype=np.uint8)
            for keypoint_part in keypoint_parts:
                cv2.fillPoly(parts_mask, [np.array(keypoint_part[0], dtype=np.int32)], (1, 1, 1))
            
            # parts_img表现=mask_new
            parts_img = parts_mask * rect_img

            height, width, _ = parts_img.shape
            bar_x = width // 3
            bar_y = height // 3
            triangle_image = parts_img[bar_y:2*bar_y, bar_x:2*bar_x]

            result, real_result, contour_list = direct.direct_detect(triangle_image)
            cv2.imwrite(output_path + name + part_name + ".png", triangle_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(output_path + name + part_name + "_result.png", result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(output_path + name + part_name + "_real_result.png", real_result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

