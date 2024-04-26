import os
import cv2
import numpy as np
import mediapipe.python.solutions as mp
import time
from scipy.interpolate import CubicSpline

# mpdraw = mp.drawing_utils
# draw_spec = mpdraw.DrawingSpec(thickness=5, circle_radius=5, color=(50,255,50))

# face_mesh为特定条件下的mp
mpface_mesh = mp.face_mesh
face_mesh = mpface_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


# from sys import getsizeof as getsize
# def get_size(var):
#     return round(getsize(var) / (1024 ** 2), 3)

def get_face_mesh(img):
    """获取nparray形式的keypoints"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_img = face_mesh.process(img_rgb)
    keypoint = []
    if result_img.multi_face_landmarks:
        # 如果检测到了，则获取第一个的所有面部特征点
        face_lms = result_img.multi_face_landmarks[0]
        # mpdraw.draw_landmarks(img, face_lms, mpface_mesh.FACEMESH_CONTOURS,
        #                       draw_spec, draw_spec)
        
        for id, lm in enumerate(face_lms.landmark):
            h, w, c = img.shape
            y, x = int(lm.y * h), int(lm.x * w)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.circle(img,(x,y),2,(50,255,50),-1)
            keypoint.append([x,y])
    return np.array(keypoint)

# from memory_profiler import profile
class CutImg:
    def __init__(self, keypoint_all):
        self.keypoint_all = keypoint_all

    def get_parts(self, img, parts):
        # 增加切割图像时，需要加入字典中
        # 传入的img是已经分割出来的人脸
        area_dict = {'right_face': 'self.right_face(img)',
                     'left_face': 'self.left_face(img)',
                     'forehead': 'self.forehead(img)',
                     'eye': 'self.eye(img)',
                     'canthus': 'self.canthus(img)',
                     'up_mouth': 'self.up_mouth(img)',
                     'jaw': 'self.jaw(img)',
                     'nose': 'self.nose(img)',
                     'below_eye': 'self.below_eye(img)',
                     'face_nose': 'self.face_nose(img)',
                     'nose_wing': 'self.nose_wing(img)',
                     'eyelash': 'self.eyelash(img)',
                     'eyebrow': 'self.eyebrow(img)',
                     'lips': 'self.lips(img)',
                     'jaw_bottom': 'self.jaw_bottom(img)',
                     'nostril': 'self.nostril(img)',
                     'between_eyebrows': 'self.between_eyebrows(img)',
                     'all_face': 'self.all_face(img)',
                     'acne_face': 'self.acne_face(img)',
                     'porphyrin_face': 'self.porphyrin_face(img)',
                     'all_face_black': 'self.all_face(img)',
                     'left_eyelash': 'self.left_eyelash(img)',
                     'right_eyelash': 'self.right_eyelash(img)',
                     'nostril_shadow': 'self.nostril_shadow(img)',
                     'left_nose_wing': 'self.left_nose_wing(img)',
                     'right_nose_wing': 'self.right_nose_wing(img)',
                     'eye_mask': 'self.eye_mask(img)',
                     'left_pore_area': 'self.left_pore_area(img)',
                     'right_pore_area':'self.right_pore_area(img)',
                     }
        
        mask0 = np.zeros(img.shape,dtype=np.uint8)
        keypoint_parts = []
        for part in parts:
            if part != "all_face" and part != "all_face_black":
                mask, keypoint_part = eval(area_dict[part])
                # mask0表示parts在img里的位置，keypoint_parts表示每个part里的关键点
                mask0[mask==1]=1
                keypoint_parts.append(keypoint_part)
            elif part == "all_face":
                # mask_0, keypoint_part_0 = eval(area_dict['eyelash'])
                # mask_1, keypoint_part_1 = eval(area_dict['eyebrow'])
                mask_2, keypoint_part_2 = eval(area_dict['lips'])
                # mask_3, keypoint_part_3 = eval(area_dict['jaw_bottom'])
                # mask_4, keypoint_part_4 = eval(area_dict['nostril'])
                # mask_5, keypoint_part_5 = eval(area_dict['forehead'])
                mask_6, keypoint_part_6 = eval(area_dict['all_face'])

                mask0[mask_6==1] = 1
                mask0[mask_2==1] = 0


                # keypoint_parts.append(keypoint_part_0)
                # keypoint_parts.append(keypoint_part_1)
                # keypoint_parts.append(keypoint_part_2)
                # keypoint_parts.append(keypoint_part_3)
                # keypoint_parts.append(keypoint_part_4)
                # keypoint_parts.append(keypoint_part_5)
                keypoint_parts.append(keypoint_part_6)

            elif part == "all_face_black":
                # mask_0, keypoint_part_0 = eval(area_dict['eyelash'])
                mask_1, keypoint_part_1 = eval(area_dict['eyebrow'])
                mask_2, keypoint_part_2 = eval(area_dict['lips'])
                mask_3, keypoint_part_3 = eval(area_dict['eye'])
                mask_4, keypoint_part_4 = eval(area_dict['nostril_shadow'])
                # mask_5, keypoint_part_5 = eval(area_dict['forehead'])
                mask_6, keypoint_part_6 = eval(area_dict['all_face'])

                mask0[mask_6==1] = 1
                mask0[mask_1==1] = 0
                mask0[mask_2==1] = 0
                mask0[mask_3==1] = 0
                mask0[mask_4==1] = 0

                # keypoint_parts.append(keypoint_part_0)
                # keypoint_parts.append(keypoint_part_1)
                # keypoint_parts.append(keypoint_part_2)
                # keypoint_parts.append(keypoint_part_3)
                # keypoint_parts.append(keypoint_part_4)
                # keypoint_parts.append(keypoint_part_5)
                keypoint_parts.append(keypoint_part_6)
        
        # mask为分割出来的人脸图像
        mask = mask0 * img
        mask_new, xyxy = self.get_min_rect(mask, keypoint_parts)
        keypoint_parts = self.get_new_keypoint_parts(xyxy, keypoint_parts)


        return mask_new, xyxy, keypoint_parts

    def get_new_keypoint_parts(self, xyxy, keypoint_parts):
        """获取在min_rect下的新的坐标点"""
        for keypoint_part in keypoint_parts:
            keypoint_part[0] -= np.array((xyxy[0], xyxy[1]))
        return keypoint_parts
    
    
    def get_min_rect(self, mask, kpp):
        """获取包含所有parts的最小矩形图片及其坐标"""
        # mask分割出来的人脸图片，kpp n * 1 * len(points)的list
        a = np.array([[0, 0]])
        bias = 20
        for i in kpp:
            for j in i:
                a = np.vstack((a, j))
        x_min = max(np.nanmin(a[1:, 0]) - bias, 0)
        y_min = max(np.nanmin(a[1:, 1]) - bias, 0)
        x_max = min(np.nanmax(a[1:, 0]) + bias, 4000)
        y_max = min(np.nanmax(a[1:, 1]) + bias, 6000)
        xyxy = [x_min, y_min, x_max, y_max]
        mask_new = mask[y_min:y_max, x_min:x_max]
        return mask_new, xyxy

    def get_soft_line(self, keypoints):
        """获取keypoints*100的平滑keypoints,保证轮廓闭合"""
        k_x = keypoints[:, 0]
        k_y = keypoints[:, 1]

        # 保证轮廓闭合
        k_x = np.append(k_x, k_x[0])
        k_y = np.append(k_y, k_y[0])

        csX = CubicSpline(np.arange(len(k_x)), k_x, bc_type='periodic')
        csY = CubicSpline(np.arange(len(k_y)), k_y, bc_type='periodic')
        IN = np.linspace(0, len(k_x) - 1, 100 * len(keypoints))
        k_x = csX(IN).reshape(-1, 1)
        k_y = csY(IN).reshape(-1, 1)
        k_all = np.hstack((k_x, k_y)).astype(np.int32)
        return k_all

    def forehead(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.07 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1.2
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.8) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.07 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)

        # 下方
        keypoint_bottom = [333, 299, 337, 108, 69, 104]
        bottom_y_bias = 0.3
        bottom_x_bias = 0.05
        for point in keypoint_bottom:
            if point in [333,104]:
                k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] + (bottom_y_bias*0.3) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] + bottom_y_bias * forehead_h]
            keypoint_new.append(k_p)

        # 单个闭合图形保持整数

        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def right_face(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_left = [135, 213, 116]
        bias = 0.03
        for point in keypoint_left:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            keypoint_new.append(k_p)
        keypoint_other = [111, 228, 229, 230, 231, 232,
                          128, 217, 126, 142, 203, 206, 212, 210]
        for point in keypoint_other:
            if point == 142:
                k_p = [self.keypoint_all[point][0] + 3*bias * (face_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1]]
                keypoint_new.append(k_p)
            else:
                keypoint_new.append(self.keypoint_all[point])
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def left_pore_area(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]
        bias = 0.03

        keypoint_right = [423, 355,343, 451, 449,]
        for point in keypoint_right:
            # k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                #    self.keypoint_all[point][1]]
            k_p = self.keypoint_all[point]
            keypoint_new.append(k_p)
            # cv2.putText(img, str(point), tuple(int(i) for i in k_p), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            # cv2.circle(img, tuple(int(i) for i in k_p), 5, (0, 0, 255), -1)

        keypoint_other = [346, 280, 427,]
        for point in keypoint_other:
            if point == 280:
                k_p = (self.keypoint_all[point][0] + 4*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)
                # k_p = self.keypoint_all[point]
                # keypoint_new.append(k_p)
            elif point == 427:
                k_p = (self.keypoint_all[point][0] + 2*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)                
            elif point == 346:
                k_p = (self.keypoint_all[point][0] - 1*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)
            else:
                k_p = self.keypoint_all[point]
                keypoint_new.append(k_p)
            # cv2.putText(img, str(point), tuple(int(i) for i in k_p), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            # cv2.circle(img, tuple(int(i) for i in k_p), 5, (0, 0, 255), -1)
        
        # 单个闭合图形保持整数
        
        # cv2.namedWindow('left_pore', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('left_pore', 400, 600)
        # cv2.imshow('left_pore', img)
        # cv2.waitKey(0)

        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def right_pore_area(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_left = [203, 126, 114, 231, 229]
        bias = 0.03
        for point in keypoint_left:
            # k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
            #        self.keypoint_all[point][1]]
            k_p = self.keypoint_all[point]
            keypoint_new.append(k_p)
            # cv2.putText(img, str(point), tuple(int(i) for i in k_p), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            # cv2.circle(img, tuple(int(i) for i in k_p), 5, (0, 0, 255), -1)
        keypoint_other = [117, 50, 207]
        for point in keypoint_other:
            if point == 50:
                k_p = (self.keypoint_all[point][0] - 4*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)
            elif point == 207:
                k_p = (self.keypoint_all[point][0] - 2*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)
            elif point == 117:
                k_p = (self.keypoint_all[point][0] + 1*bias * abs((face_center_x - self.keypoint_all[point][0])), self.keypoint_all[point][1])        
                keypoint_new.append(k_p)
            else:
                k_p = self.keypoint_all[point]
                keypoint_new.append(k_p)
            # cv2.putText(img, str(point), tuple(int(i) for i in k_p), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            # cv2.circle(img, tuple(int(i) for i in k_p), 5, (0, 0, 255), -1)


        # 单个闭合图形保持整数
            
        # cv2.namedWindow('right_pore', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('right_pore', 400, 600)
        # cv2.imshow('right_pore', img)
        # cv2.waitKey(0)

        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def left_face(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_right = [345, 433, 364]
        bias = 0.03
        for point in keypoint_right:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            keypoint_new.append(k_p)
        keypoint_other = [430, 432, 426, 423, 371, 355, 437,
                          357, 452, 451, 450, 449, 448, 340]
        for point in keypoint_other:
            if point == 371:
                k_p = [self.keypoint_all[point][0] + 3*bias * (face_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1]]
                keypoint_new.append(k_p)
            else:
                keypoint_new.append(self.keypoint_all[point])
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def eye(self, img):
        left_eye = self.keypoint_all[[414, 286, 258, 257, 259, 260, 467, 446,
                                      261, 448, 449, 450, 451, 452, 453, 464]]
        right_eye = self.keypoint_all[[226, 247, 30, 29, 27, 28, 56, 190,
                                       244, 233, 232, 231, 230, 229, 228, 31]]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left_eye = self.get_soft_line(left_eye)
        right_eye = self.get_soft_line(right_eye)
        cv2.fillPoly(mask, [left_eye, right_eye], (1))
        return mask, [left_eye, right_eye]

    def canthus(self, img):
        left_canthus = self.keypoint_all[[359, 342, 353, 383, 372, 345, 340, 261]]
        right_canthus = self.keypoint_all[[156, 124, 113, 130, 31, 111, 116, 143]]

        left_canthus = self.get_soft_line(left_canthus)
        right_canthus = self.get_soft_line(right_canthus)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_canthus, right_canthus], (1))
        return mask, [left_canthus, right_canthus]

    def up_mouth(self, img):
        up_mouth_h = self.keypoint_all[0][1] - self.keypoint_all[2][1]
        keypoint_top = self.keypoint_all[[98, 97, 2, 326, 327]]
        keypoint_top[:, 1] = keypoint_top[:, 1] + 0.1 * up_mouth_h

        keypoint_bottom = self.keypoint_all[[410, 269, 267, 37, 39, 186]]
        keypoint_bottom[:, 1] = keypoint_bottom[:, 1] - 0.2 * up_mouth_h
        keypoint_new = np.vstack((keypoint_top, keypoint_bottom))
        keypoint_new = self.get_soft_line(keypoint_new)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def jaw(self, img):
        # 作为偏置项bias
        jaw_h = self.keypoint_all[199][1] - self.keypoint_all[17][1]
        keypoint_top = self.keypoint_all[[210, 202, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 422, 430]]
        keypoint_top[:, 1] = keypoint_top[:, 1] + 0.1 * jaw_h

        keypoint_bottom = self.keypoint_all[[394, 395, 262, 421, 200, 201, 32, 170, 169]]
        keypoint_bottom[:, 1] = keypoint_bottom[:, 1] + 0.1 * jaw_h
        keypoint_new = np.vstack((keypoint_top, keypoint_bottom))
        keypoint_new = self.get_soft_line(keypoint_new)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))

        mask_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask[mask_gray < 50] = 0

        return mask, [keypoint_new]

    def nose(self, img):
        nose0 = self.keypoint_all[[193, 168, 417, 351, 419, 456, 360, 278,
                                   439, 289]]
        nose_bias_point = self.keypoint_all[[440, 275, 370, 141, 45, 220]]
        nose_bias_point[[0, 1, 4, 5], 1] = nose_bias_point[[0, 1, 4, 5], 1] * 1.018
        nose_bias_point[[2, 3], 1] = nose_bias_point[[2, 3], 1] * 0.99
        nose1 = self.keypoint_all[[59, 219, 48, 131, 236, 196,
                                   122]]
        nose = np.vstack((nose0, nose_bias_point, nose1))
        nose = np.array(nose).astype(np.int32)
        nose = self.get_soft_line(nose)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [nose], (1))

        return mask, [nose]

    def below_eye(self, img):
        left_be_top = self.keypoint_all[[453, 463, 341, 256, 252, 253, 254, 339, 255, 261]]
        left_be_bottom = self.keypoint_all[[448, 449, 450, 451, 452]]
        right_be_top = self.keypoint_all[[31, 25, 110, 24, 23, 22, 26, 112, 243, 233]]
        right_be_bottom = self.keypoint_all[[232, 231, 230, 229, 228]]

        left_be_top[:, 1] = left_be_top[:, 1] * 1.01
        right_be_top[:, 1] = right_be_top[:, 1] * 1.01

        left_be_bottom[:1, 1] = left_be_bottom[:1, 1] * 1.05
        left_be_bottom[1:, 1] = left_be_bottom[1:, 1] * 1.07

        right_be_bottom[:-1, 1] = right_be_bottom[:-1, 1] * 1.07
        right_be_bottom[-1:, 1] = right_be_bottom[-1:, 1] * 1.05

        left_be = np.vstack((left_be_top, left_be_bottom))
        right_be = np.vstack((right_be_top, right_be_bottom))
        left_be = self.get_soft_line(left_be)
        right_be = self.get_soft_line(right_be)

        # left_be = np.vstack((left_be_top, left_be_bottom))
        # right_be = np.vstack((right_be_top, right_be_bottom))
        # left_be = self.get_soft_line(left_be)
        # right_be = self.get_soft_line(right_be)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_be, right_be], (1))

        return mask, [left_be, right_be]

    def face_nose(self, img):
        fn_top = self.keypoint_all[[111, 31, 228, 229, 230, 231, 232, 233,
                                    245, 6, 465, 453, 452, 451, 450, 449, 448,
                                    261, 340]]

        fn_left = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_right = [345, 352, 376, 433, 364]
        bias = 0.03
        for point in keypoint_right:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            fn_left.append(k_p)

        fn_bottom = self.keypoint_all[[430, 432, 436, 426, 423, 278, 344,
                                       440, 275, 45, 220, 115, 48,
                                       203, 206, 216, 212, 210]]
        fn_bottom[:, 1] = fn_bottom[:, 1] + 40

        fn_right = []
        keypoint_left = [135, 213, 147, 123, 116]
        bias = 0.03
        for point in keypoint_left:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            fn_right.append(k_p)

        fn = np.vstack((fn_top, fn_left, fn_bottom, fn_right))
        fn = np.array(fn).astype(np.int32)
        fn = self.get_soft_line(fn)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fn], (1))
        return mask, [fn]

    def nose_wing(self, img):

        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        left_nosewing = self.keypoint_all[[425, 427, 434, 430, 395, 262, 418, 335, 273, 410, 322, 327, 294, 358]]
        left_nosewing[0][0] = left_nosewing[0][0] - 3 * bias
        left_nosewing[1][0] = left_nosewing[1][0] - 2 * bias
        left_nosewing[2][0] = left_nosewing[2][0] - bias
        left_nosewing[-2][0] = left_nosewing[-2][0] + bias
        left_nosewing[-3][0] = left_nosewing[-3][0] + bias
        right_nosewing = self.keypoint_all[[129, 64, 98, 92, 186, 43, 106, 194, 32, 170, 210, 214, 207, 205]]
        right_nosewing[-1][0] = right_nosewing[-1][0] + 3 * bias
        right_nosewing[-2][0] = right_nosewing[-2][0] + 2 * bias
        right_nosewing[-3][0] = right_nosewing[-3][0] + bias
        right_nosewing[1][0] = right_nosewing[1][0] - bias
        right_nosewing[2][0] = right_nosewing[2][0] - bias
        # left_nosewing[:, 0] = left_nosewing[:, 0] + 1 * bias
        # right_nosewing[:, 0] = right_nosewing[:, 0] - 1 * bias
        left_nosewing = self.get_soft_line(left_nosewing)
        right_nosewing = self.get_soft_line(right_nosewing)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_nosewing, right_nosewing], (1))
        return mask, [left_nosewing, right_nosewing]

    def left_nose_wing(self, img):
        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        left_nosewing = self.keypoint_all[[425, 427, 434, 430, 395, 262, 418, 335, 273, 410, 322, 327, 294, 358]]
        left_nosewing[0][0] = left_nosewing[0][0] - 3 * bias
        left_nosewing[1][0] = left_nosewing[1][0] - 2 * bias
        left_nosewing[2][0] = left_nosewing[2][0] - bias
        left_nosewing[-2][0] = left_nosewing[-2][0] + bias
        left_nosewing[-3][0] = left_nosewing[-3][0] + bias
        left_nosewing = self.get_soft_line(left_nosewing)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_nosewing], (1))
        return mask, [left_nosewing]

    def right_nose_wing(self, img):
        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        right_nosewing = self.keypoint_all[[129, 64, 98, 92, 186, 43, 106, 194, 32, 170, 210, 214, 207, 205]]
        right_nosewing[-1][0] = right_nosewing[-1][0] + 3 * bias
        right_nosewing[-2][0] = right_nosewing[-2][0] + 2 * bias
        right_nosewing[-3][0] = right_nosewing[-3][0] + bias
        right_nosewing[1][0] = right_nosewing[1][0] - bias
        right_nosewing[2][0] = right_nosewing[2][0] - bias
        right_nosewing = self.get_soft_line(right_nosewing)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [right_nosewing], (1))
        return mask, [right_nosewing]

    def eyelash(self, img):
        left_eye_center_y = (self.keypoint_all[450][1] - self.keypoint_all[257][1]) / 2 + \
                            self.keypoint_all[257][1]
        right_eye_center_y = (self.keypoint_all[230][1] - self.keypoint_all[27][1]) / 2 + \
                             self.keypoint_all[27][1]

        left_eye_top = self.keypoint_all[[414, 286, 258, 257, 259]]
        left_eye_top[[0], 1] = left_eye_top[[0], 1] + 0.1 * (left_eye_center_y - left_eye_top[[0], 1])
        left_eye_top[1:, 1] = left_eye_top[1:, 1] + 0.3 * (left_eye_center_y - left_eye_top[1:, 1])
        left_eye_top[[2, 3], 1] = left_eye_top[[2, 3], 1] + 0.2 * (left_eye_center_y - left_eye_top[[2, 3], 1])
        left_eye_top = np.vstack((left_eye_top, self.keypoint_all[[359]]))
        left_eye_bottom = self.keypoint_all[[261, 449, 450, 451, 453]]
        left_eye_bottom[:, 1] = left_eye_bottom[:, 1] - 0 * (left_eye_bottom[:, 1] - left_eye_center_y)
        left_eye = np.vstack((left_eye_top, left_eye_bottom))

        right_eye_top = self.keypoint_all[[29, 27, 28, 56, 190]]
        right_eye_top[[-1], 1] = right_eye_top[[-1], 1] + 0.1 * (right_eye_center_y - right_eye_top[[-1], 1])
        right_eye_top[:-1, 1] = right_eye_top[:-1, 1] + 0.3 * (right_eye_center_y - right_eye_top[:-1, 1])
        right_eye_top[[1, 2], 1] = right_eye_top[[1, 2], 1] + 0.2 * (right_eye_center_y - right_eye_top[[1, 2], 1])
        right_eye_top = np.vstack((self.keypoint_all[[130]], right_eye_top))
        right_eye_bottom = self.keypoint_all[[233, 231, 230, 229, 31]]
        right_eye_bottom[:, 1] = right_eye_bottom[:, 1] - 0 * (right_eye_bottom[:, 1] - right_eye_center_y)
        right_eye = np.vstack((right_eye_top, right_eye_bottom))

        left_eye = self.get_soft_line(left_eye)
        right_eye = self.get_soft_line(right_eye)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_eye, right_eye], (1))
        return mask, [left_eye, right_eye]

    def left_eyelash(self, img):
        left_eye_center_y = (self.keypoint_all[450][1] - self.keypoint_all[257][1]) / 2 + \
                            self.keypoint_all[257][1]

        left_eye_top = self.keypoint_all[[414, 286, 258, 257, 259]]
        left_eye_top[[0], 1] = left_eye_top[[0], 1] + 0.1 * (left_eye_center_y - left_eye_top[[0], 1])
        left_eye_top[1:, 1] = left_eye_top[1:, 1] + 0.3 * (left_eye_center_y - left_eye_top[1:, 1])
        left_eye_top[[2, 3], 1] = left_eye_top[[2, 3], 1] + 0.2 * (left_eye_center_y - left_eye_top[[2, 3], 1])
        left_eye_top = np.vstack((left_eye_top, self.keypoint_all[[359]]))
        left_eye_bottom = self.keypoint_all[[261, 449, 450, 451, 453]]
        left_eye_bottom[:, 1] = left_eye_bottom[:, 1] - 0 * (left_eye_bottom[:, 1] - left_eye_center_y)
        left_eye = np.vstack((left_eye_top, left_eye_bottom))

        left_eye = self.get_soft_line(left_eye)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_eye], (1))
        return mask, [left_eye]

    def right_eyelash(self, img):
        right_eye_center_y = (self.keypoint_all[230][1] - self.keypoint_all[27][1]) / 2 + \
                             self.keypoint_all[27][1]
        right_eye_top = self.keypoint_all[[29, 27, 28, 56, 190]]
        right_eye_top[[-1], 1] = right_eye_top[[-1], 1] + 0.1 * (right_eye_center_y - right_eye_top[[-1], 1])
        right_eye_top[:-1, 1] = right_eye_top[:-1, 1] + 0.3 * (right_eye_center_y - right_eye_top[:-1, 1])
        right_eye_top[[1, 2], 1] = right_eye_top[[1, 2], 1] + 0.2 * (right_eye_center_y - right_eye_top[[1, 2], 1])
        right_eye_top = np.vstack((self.keypoint_all[[130]], right_eye_top))
        right_eye_bottom = self.keypoint_all[[233, 231, 230, 229, 31]]
        right_eye_bottom[:, 1] = right_eye_bottom[:, 1] - 0 * (right_eye_bottom[:, 1] - right_eye_center_y)
        right_eye = np.vstack((right_eye_top, right_eye_bottom))


        right_eye = self.get_soft_line(right_eye)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [right_eye], (1))
        return mask, [right_eye]


        # 眉毛

    def eyebrow(self, img):
        left_eb_h = self.keypoint_all[285][1] - self.keypoint_all[336][1]
        left_eb_top = self.keypoint_all[[336, 334, 300]]
        left_eb_top[:, 1] = left_eb_top[:, 1] - 0.6 * left_eb_h
        left_eb_bottom = self.keypoint_all[[276, 282, 285]]
        left_eb_bottom[:, 1] = left_eb_bottom[:, 1] + 0.6 * left_eb_h
        left_eb = np.vstack((left_eb_top, left_eb_bottom))
        left_eb[[0, 5], 0] = left_eb[[0, 5], 0] - 30
        left_eb[[2, 3], 0] = left_eb[[2, 3], 0] + 50

        right_eb_h = self.keypoint_all[55][1] - self.keypoint_all[107][1]
        right_eb_top = self.keypoint_all[[70, 105, 107]]
        right_eb_top[:, 1] = right_eb_top[:, 1] - 0.6 * right_eb_h
        right_eb_bottom = self.keypoint_all[[55, 52, 46]]
        right_eb_bottom[:, 1] = right_eb_bottom[:, 1] + 0.6 * right_eb_h
        right_eb = np.vstack((right_eb_top, right_eb_bottom))
        right_eb[[0, 5], 0] = right_eb[[0, 5], 0] - 50
        right_eb[[2, 3], 0] = right_eb[[2, 3], 0] + 30

        left_eb = self.get_soft_line(left_eb)
        right_eb = self.get_soft_line(right_eb)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [left_eb, right_eb], (1))
        return mask, [left_eb, right_eb]

        # 嘴唇

    def lips(self, img):
        lips_top = self.keypoint_all[[185, 39, 37, 267, 269, 409]]
        lips_top[:, 1] = lips_top[:, 1] - 80
        lips_61 = [self.keypoint_all[61][0] - 80, self.keypoint_all[61][1]]
        lips_291 = [self.keypoint_all[291][0] + 80, self.keypoint_all[291][1]]

        lips_top = np.vstack((lips_61, lips_top, lips_291))
        lips_bottom = self.keypoint_all[[321, 405, 314, 17, 84, 181, 91]]
        lips_bottom[:, 1] = lips_bottom[:, 1] + 50

        lips = np.vstack((lips_top, lips_bottom))
        lips = self.get_soft_line(lips)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [lips], (1))
        return mask, [lips]

        # 下巴下面托盘

    def jaw_bottom(self, img):
        jaw_bottom = self.keypoint_all[[149, 32, 201, 200, 421, 262, 378, 400, 377, 152, 148, 176]]

        jaw_bottom = self.get_soft_line(jaw_bottom)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [jaw_bottom], (1))
        return mask, [jaw_bottom]

        # 鼻孔

    def nostril(self, img):
        nostril_top = self.keypoint_all[[79, 44, 19, 274, 309]]
        nostril_top[:, 1] = nostril_top[:, 1] - 30
        nostril_bottom = self.keypoint_all[[460, 2, 98]]
        nostril = np.vstack((nostril_top, nostril_bottom))

        nostril = self.get_soft_line(nostril)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [nostril], (1))
        return mask, [nostril]

    def nostril_shadow(self, img):
        nostril_top = self.keypoint_all[[79, 44, 19, 274, 309]]
        nostril_top[2][1] = nostril_top[2][1] - 30
        nose_shadow_left2 = self.keypoint_all[[358, 429]]
        nose_shadow_left1 = self.keypoint_all[[429, 358, 327]]
        nose_shadow_left1[:, 0] = nose_shadow_left1[:, 0] + 45
        nose_shadow_left2[:, 0] = nose_shadow_left2[:, 0] - 130
        nostril_top[:, 1] = nostril_top[:, 1] - 30

        nostril_bottom = self.keypoint_all[[2]]
        nose_shadow_right2 = self.keypoint_all[[98, 129, 209]]
        nose_shadow_right1 = self.keypoint_all[[209, 129]]
        nose_shadow_right1[:, 0] = nose_shadow_right1[:, 0] + 130
        nose_shadow_right2[:, 0] = nose_shadow_right2[:, 0] - 45

        nostril_shadow = np.vstack(
            (nostril_top, nose_shadow_left2, nose_shadow_left1, nostril_bottom, nose_shadow_right2, nose_shadow_right1))

        nostril_shadow = self.get_soft_line(nostril_shadow)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [nostril_shadow], (1))
        return mask, [nostril_shadow]

    def between_eyebrows(self, img):
        bias = 81
        betweeen_eyebrows = self.keypoint_all[[151, 337, 336, 285, 168, 55, 107, 108]]
        betweeen_eyebrows[2][0] = betweeen_eyebrows[2][0] - 1 * bias
        betweeen_eyebrows[3][0] = betweeen_eyebrows[3][0] - 1 * bias
        betweeen_eyebrows[-2][0] = betweeen_eyebrows[-2][0] + 1 * bias
        betweeen_eyebrows[-3][0] = betweeen_eyebrows[-3][0] + 1 * bias
        betweeen_eyebrows = self.get_soft_line(betweeen_eyebrows)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [betweeen_eyebrows], (1))
        return mask, [betweeen_eyebrows]


    def all_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)


        other = self.keypoint_all[[300,345,433,364,395,
                         262,200,32,170,135,213,116,70]]
        keypoint_new = np.vstack((keypoint_new,other))
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask,[keypoint_new]

    def acne_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)


        other = self.keypoint_all[[300,444,442,413,453,449,

                        345,433,364,395,
                         262,200,32,170,135,213,116,
                         229,233,189,222,224,70

                                   ]]
        keypoint_new = np.vstack((keypoint_new,other))

        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask,[keypoint_new]

    def porphyrin_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)

        # 下方
        keypoint_bottom = [298, 333, 299, 69, 104, 68]
        bottom_y_bias = 0.4
        bottom_x_bias = 0.05
        for point in keypoint_bottom:
            k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1] + bottom_y_bias * forehead_h]
            keypoint_new.append(k_p)

        other = self.keypoint_all[[336,465,451,449,346,352,
                                   416,430,422,410,393,167,186,202,
                                   210,192,123,117,229,231,245,
                                   107]]
        other[[0],0] = other[[0],0]-40
        other[[4],1] = other[[4],1]-40
        other[[-1],0] = other[[-1],0]+40
        other[[-5],1] = other[[-5],1]-40
        other[[0,-1],1] = other[[0,-1],1]+40
        keypoint_new = np.array(keypoint_new)
        keypoint_new = np.insert(keypoint_new, -3, other, axis=0)
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)

        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        return mask, [keypoint_new]

    def eye_mask(self, img):
        key_list = self.keypoint_all[[446, 261, 448, 449, 450, 451, 452, 6,
                                      232, 231, 230, 229, 228, 31, 226,
                                      113, 225, 222, 442, 445, 342]]

        key_list = self.get_soft_line(key_list)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [key_list], (1))
        return mask, [key_list]
    
    def get_area(self, size):
        center_right = (self.keypoint_all[36][0]-150, self.keypoint_all[36][1]-150)
        center_left = (self.keypoint_all[266][0]+150, self.keypoint_all[266][1]-150)
        right_points = ((center_right[0] - size, center_right[1] - size), (center_right[0] + size, center_right[1] + size))
        left_points = ((center_left[0] - size, center_left[1] - size), (center_left[0] + size, center_left[1] + size))
        return right_points, left_points

def usm(img, w=0.3):  # w影响较大，0.3较为合适
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    a = 1 / (1 - w)
    b = -w / (1 - w)
    usm_out = cv2.addWeighted(img, a, blur_img, b, 0)
    return usm_out


class CutImg1:
    def __init__(self, keypoint_all):
        self.keypoint_all = keypoint_all

    def get_parts(self, img, parts):
        # 增加切割图像时，需要加入字典中

        area_dict = {'right_face': 'self.right_face(img)',
                     'left_face': 'self.left_face(img)',
                     'forehead': 'self.forehead(img)',
                     'eye': 'self.eye(img)',
                     'canthus': 'self.canthus(img)',
                     'up_mouth': 'self.up_mouth(img)',
                     'jaw': 'self.jaw(img)',
                     'nose': 'self.nose(img)',
                     'below_eye': 'self.below_eye(img)',
                     'face_nose': 'self.face_nose(img)',
                     'nose_wing': 'self.nose_wing(img)',
                     'eyelash': 'self.eyelash(img)',
                     'eyebrow': 'self.eyebrow(img)',
                     'lips': 'self.lips(img)',
                     'jaw_bottom': 'self.jaw_bottom(img)',
                     'nostril': 'self.nostril(img)',
                     'between_eyebrows': 'self.between_eyebrows(img)',
                     'all_face': 'self.all_face(img)',
                     'acne_face': 'self.acne_face(img)',
                     'porphyrin_face': 'self.porphyrin_face(img)',
                     'all_face_black': 'self.all_face(img)',
                     'left_eyelash': 'self.left_eyelash(img)',
                     'right_eyelash': 'self.right_eyelash(img)',
                     'nostril_shadow': 'self.nostril_shadow(img)',
                     'left_nose_wing': 'self.left_nose_wing(img)',
                     'right_nose_wing': 'self.right_nose_wing(img)',
                     'eye_mask': 'self.eye_mask(img)',
                     }
        mask0 = np.zeros(img.shape, dtype=np.uint8)
        keypoint_parts = []
        for part in parts:
            if part != "all_face" and part != "all_face_black":
                mask, keypoint_part = eval(area_dict[part])
                mask0[mask==1] = 1
                keypoint_parts.append(keypoint_part)
            elif part == "all_face":
                # mask_0, keypoint_part_0 = eval(area_dict['eyelash'])
                # mask_1, keypoint_part_1 = eval(area_dict['eyebrow'])
                mask_2, keypoint_part_2 = eval(area_dict['lips'])
                # mask_3, keypoint_part_3 = eval(area_dict['jaw_bottom'])
                # mask_4, keypoint_part_4 = eval(area_dict['nostril'])
                # mask_5, keypoint_part_5 = eval(area_dict['forehead'])
                mask_6, keypoint_part_6 = eval(area_dict['all_face'])

                mask0 = mask_2.astype(np.int16)

                mask1 = mask_6.astype(np.int16)

                # keypoint_parts.append(keypoint_part_0)
                # keypoint_parts.append(keypoint_part_1)
                # keypoint_parts.append(keypoint_part_2)
                # keypoint_parts.append(keypoint_part_3)
                # keypoint_parts.append(keypoint_part_4)
                # keypoint_parts.append(keypoint_part_5)
                keypoint_parts.append(keypoint_part_6)
                mask0 = 1 - np.array(mask0, dtype=bool)
                mask0 = mask0 * mask1

            elif part == "all_face_black":
                # mask_0, keypoint_part_0 = eval(area_dict['eyelash'])
                mask_1, keypoint_part_1 = eval(area_dict['eyebrow'])
                mask_2, keypoint_part_2 = eval(area_dict['lips'])
                mask_3, keypoint_part_3 = eval(area_dict['eye'])
                mask_4, keypoint_part_4 = eval(area_dict['nostril_shadow'])
                # mask_5, keypoint_part_5 = eval(area_dict['forehead'])
                mask_6, keypoint_part_6 = eval(area_dict['all_face'])

                mask0 = mask_1.astype(np.int16)+mask_2.astype(np.int16)+\
                        mask_3.astype(np.int16)+mask_4.astype(np.int16)

                mask1 = mask_6.astype(np.int16)

                # keypoint_parts.append(keypoint_part_0)
                # keypoint_parts.append(keypoint_part_1)
                # keypoint_parts.append(keypoint_part_2)
                # keypoint_parts.append(keypoint_part_3)
                # keypoint_parts.append(keypoint_part_4)
                # keypoint_parts.append(keypoint_part_5)
                keypoint_parts.append(keypoint_part_6)
                mask0 = 1 - np.array(mask0, dtype=bool)
                mask0 = mask0 * mask1

        mask0 = np.array(mask0, dtype=bool)
        mask = mask0 * img
        mask_new, xyxy = self.get_min_rect(mask, keypoint_parts)
        return mask_new, xyxy, keypoint_parts

    def get_min_rect(self, mask, kpp):
        a = np.array([[0, 0]])
        bias = 20
        for i in kpp:
            for j in i:
                a = np.vstack((a, j))
        x_min = max(np.nanmin(a[1:, 0]) - bias, 0)
        y_min = max(np.nanmin(a[1:, 1]) - bias, 0)
        x_max = min(np.nanmax(a[1:, 0]) + bias, 4000)
        y_max = min(np.nanmax(a[1:, 1]) + bias, 6000)
        xyxy = [x_min, y_min, x_max, y_max]
        mask_new = mask[y_min:y_max, x_min:x_max]
        return mask_new, xyxy

    def get_soft_line(self, keypoints):
        k_x = keypoints[:, 0]
        k_y = keypoints[:, 1]

        k_x = np.append(k_x, k_x[0])
        k_y = np.append(k_y, k_y[0])

        csX = CubicSpline(np.arange(len(k_x)), k_x, bc_type='periodic')
        csY = CubicSpline(np.arange(len(k_y)), k_y, bc_type='periodic')
        IN = np.linspace(0, len(k_x) - 1, 1000 * len(keypoints))
        k_x = csX(IN).reshape(-1, 1)
        k_y = csY(IN).reshape(-1, 1)
        k_all = np.hstack((k_x, k_y)).astype(np.int32)
        return k_all

    def forehead(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.07 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1.2
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.8) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.07 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)

        # 下方
        keypoint_bottom = [333, 299, 337, 108, 69, 104]
        bottom_y_bias = 0.3
        bottom_x_bias = 0.05
        for point in keypoint_bottom:
            if point in [333,104]:
                k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] + (bottom_y_bias*0.3) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] + bottom_y_bias * forehead_h]
            keypoint_new.append(k_p)

        # 单个闭合图形保持整数

        keypoint_new = np.array(keypoint_new).astype(np.int32)

        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1))
        # mask = img * mask
        # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask[mask_gray < 30] = 0
        return mask, [keypoint_new]

    def right_face(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_left = [135, 213, 116]
        bias = 0.03
        for point in keypoint_left:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            keypoint_new.append(k_p)
        keypoint_other = [111, 228, 229, 230, 231, 232,
                          128, 217, 126, 142, 203, 206, 212, 210]
        for point in keypoint_other:
            if point == 142:
                k_p = [self.keypoint_all[point][0] + 3*bias * (face_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1]]
                keypoint_new.append(k_p)
            else:
                keypoint_new.append(self.keypoint_all[point])
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        return mask, [keypoint_new]

    def left_face(self, img):
        keypoint_new = []
        face_center_x = self.keypoint_all[5][0]

        
        keypoint_right = [345, 433, 364]
        bias = 0.03
        for point in keypoint_right:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            keypoint_new.append(k_p)
        

        keypoint_other = [430, 432, 426, 423, 371, 355, 437,
                          357, 452, 451, 450, 449, 448, 340]
        for point in keypoint_other:
            if point == 371:
                k_p = [self.keypoint_all[point][0] + 3*bias * (face_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1]]
                keypoint_new.append(k_p)
            else:
                keypoint_new.append(self.keypoint_all[point])
        
        
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)
        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        return mask, [keypoint_new]

    def eye(self, img):
        left_eye = self.keypoint_all[[414, 286, 258, 257, 259, 260, 467, 446,
                                      261, 448, 449, 450, 451, 452, 453, 464]]
        right_eye = self.keypoint_all[[226, 247, 30, 29, 27, 28, 56, 190,
                                       244, 233, 232, 231, 230, 229, 228, 31]]
        mask = np.zeros(img.shape, dtype=np.uint8)
        left_eye = self.get_soft_line(left_eye)
        right_eye = self.get_soft_line(right_eye)
        cv2.fillPoly(mask, [left_eye, right_eye], (1, 1, 1))
        mask = img * mask
        return mask, [left_eye, right_eye]

    def canthus(self, img):
        left_canthus = self.keypoint_all[[359, 342, 353, 383, 372, 345, 340, 261]]
        right_canthus = self.keypoint_all[[156, 124, 113, 130, 31, 111, 116, 143]]
        mask = np.zeros(img.shape, dtype=np.uint8)
        left_canthus = self.get_soft_line(left_canthus)
        right_canthus = self.get_soft_line(right_canthus)
        cv2.fillPoly(mask, [left_canthus, right_canthus], (1, 1, 1))
        mask = img * mask
        return mask, [left_canthus, right_canthus]

    def up_mouth(self, img):
        up_mouth_h = self.keypoint_all[0][1] - self.keypoint_all[2][1]
        keypoint_top = self.keypoint_all[[98, 97, 2, 326, 327]]
        keypoint_top[:, 1] = keypoint_top[:, 1] + 0.1 * up_mouth_h

        keypoint_bottom = self.keypoint_all[[410, 269, 267, 37, 39, 186]]
        keypoint_bottom[:, 1] = keypoint_bottom[:, 1] - 0.2 * up_mouth_h
        keypoint_new = np.vstack((keypoint_top, keypoint_bottom))
        keypoint_new = self.get_soft_line(keypoint_new)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        return mask, [keypoint_new]

    def jaw(self, img):
        # 作为偏置项bias
        jaw_h = self.keypoint_all[199][1] - self.keypoint_all[17][1]
        keypoint_top = self.keypoint_all[[210, 202, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 422, 430]]
        keypoint_top[:, 1] = keypoint_top[:, 1] + 0.1 * jaw_h

        keypoint_bottom = self.keypoint_all[[394, 395, 262, 421, 200, 201, 32, 170, 169]]
        keypoint_bottom[:, 1] = keypoint_bottom[:, 1] + 0.1 * jaw_h
        keypoint_new = np.vstack((keypoint_top, keypoint_bottom))
        keypoint_new = self.get_soft_line(keypoint_new)

        keypoint_new = self.get_soft_line(keypoint_new)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))

        mask = img * mask
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask_gray < 50] = 0

        return mask, [keypoint_new]

    def nose(self, img):
        nose0 = self.keypoint_all[[193, 168, 417, 351, 419, 456, 360, 278,
                                   439, 289]]
        nose_bias_point = self.keypoint_all[[440, 275, 370, 141, 45, 220]]
        nose_bias_point[[0, 1, 4, 5], 1] = nose_bias_point[[0, 1, 4, 5], 1] * 1.018
        nose_bias_point[[2, 3], 1] = nose_bias_point[[2, 3], 1] * 0.99
        nose1 = self.keypoint_all[[59, 219, 48, 131, 236, 196,
                                   122]]
        nose = np.vstack((nose0, nose_bias_point, nose1))
        nose = np.array(nose).astype(np.int32)
        nose = self.get_soft_line(nose)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [nose], (1, 1, 1))
        mask = img * mask

        return mask, [nose]

    def below_eye(self, img):
        left_be_top = self.keypoint_all[[453, 463, 341, 256, 252, 253, 254, 339, 255, 261]]
        left_be_bottom = self.keypoint_all[[448, 449, 450, 451, 452]]
        right_be_top = self.keypoint_all[[31, 25, 110, 24, 23, 22, 26, 112, 243, 233]]
        right_be_bottom = self.keypoint_all[[232, 231, 230, 229, 228]]

        left_be_top[:, 1] = left_be_top[:, 1] * 1.01
        right_be_top[:, 1] = right_be_top[:, 1] * 1.01

        left_be_bottom[:1, 1] = left_be_bottom[:1, 1] * 1.05
        left_be_bottom[1:, 1] = left_be_bottom[1:, 1] * 1.07

        right_be_bottom[:-1, 1] = right_be_bottom[:-1, 1] * 1.07
        right_be_bottom[-1:, 1] = right_be_bottom[-1:, 1] * 1.05

        left_be = np.vstack((left_be_top, left_be_bottom))
        right_be = np.vstack((right_be_top, right_be_bottom))
        left_be = self.get_soft_line(left_be)
        right_be = self.get_soft_line(right_be)

        # left_be = np.vstack((left_be_top, left_be_bottom))
        # right_be = np.vstack((right_be_top, right_be_bottom))
        # left_be = self.get_soft_line(left_be)
        # right_be = self.get_soft_line(right_be)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [left_be, right_be], (1, 1, 1))
        mask = img * mask

        return mask, [left_be, right_be]

    def face_nose(self, img):
        fn_top = self.keypoint_all[[111, 31, 228, 229, 230, 231, 232, 233,
                                    245, 6, 465, 453, 452, 451, 450, 449, 448,
                                    261, 340]]

        fn_left = []
        face_center_x = self.keypoint_all[5][0]
        keypoint_right = [345, 352, 376, 433, 364]
        bias = 0.03
        for point in keypoint_right:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            fn_left.append(k_p)

        fn_bottom = self.keypoint_all[[430, 432, 436, 426, 423, 278, 344,
                                       440, 275, 45, 220, 115, 48,
                                       203, 206, 216, 212, 210]]
        fn_bottom[:, 1] = fn_bottom[:, 1] + 40

        fn_right = []
        keypoint_left = [135, 213, 147, 123, 116]
        bias = 0.03
        for point in keypoint_left:
            k_p = [self.keypoint_all[point][0] + bias * (face_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1]]
            fn_right.append(k_p)

        fn = np.vstack((fn_top, fn_left, fn_bottom, fn_right))
        fn = np.array(fn).astype(np.int32)
        fn = self.get_soft_line(fn)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [fn], (1, 1, 1))
        mask = img * mask
        return mask, [fn]

    def nose_wing(self, img):

        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        left_nosewing = self.keypoint_all[[425, 427, 434, 430, 395, 262, 418, 335, 273, 410, 322, 327, 294, 358]]
        left_nosewing[0][0] = left_nosewing[0][0] - 3 * bias
        left_nosewing[1][0] = left_nosewing[1][0] - 2 * bias
        left_nosewing[2][0] = left_nosewing[2][0] - bias
        left_nosewing[-2][0] = left_nosewing[-2][0] + bias
        left_nosewing[-3][0] = left_nosewing[-3][0] + bias
        right_nosewing = self.keypoint_all[[129, 64, 98, 92, 186, 43, 106, 194, 32, 170, 210, 214, 207, 205]]
        right_nosewing[-1][0] = right_nosewing[-1][0] + 3 * bias
        right_nosewing[-2][0] = right_nosewing[-2][0] + 2 * bias
        right_nosewing[-3][0] = right_nosewing[-3][0] + bias
        right_nosewing[1][0] = right_nosewing[1][0] - bias
        right_nosewing[2][0] = right_nosewing[2][0] - bias
        # left_nosewing[:, 0] = left_nosewing[:, 0] + 1 * bias
        # right_nosewing[:, 0] = right_nosewing[:, 0] - 1 * bias
        left_nosewing = self.get_soft_line(left_nosewing)
        right_nosewing = self.get_soft_line(right_nosewing)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [left_nosewing, right_nosewing], (1, 1, 1))
        mask = img * mask
        return mask, [left_nosewing, right_nosewing]

    def left_nose_wing(self, img):
        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        left_nosewing = self.keypoint_all[[425, 427, 434, 430, 395, 262, 418, 335, 273, 410, 322, 327, 294, 358]]
        left_nosewing[0][0] = left_nosewing[0][0] - 3 * bias
        left_nosewing[1][0] = left_nosewing[1][0] - 2 * bias
        left_nosewing[2][0] = left_nosewing[2][0] - bias
        left_nosewing[-2][0] = left_nosewing[-2][0] + bias
        left_nosewing[-3][0] = left_nosewing[-3][0] + bias
        left_nosewing = self.get_soft_line(left_nosewing)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [left_nosewing], (1, 1, 1))
        mask = img * mask
        return mask, [left_nosewing]

    def right_nose_wing(self, img):
        bias = 0.1 * (self.keypoint_all[98][0] - self.keypoint_all[214][0])
        right_nosewing = self.keypoint_all[[129, 64, 98, 92, 186, 43, 106, 194, 32, 170, 210, 214, 207, 205]]
        right_nosewing[-1][0] = right_nosewing[-1][0] + 3 * bias
        right_nosewing[-2][0] = right_nosewing[-2][0] + 2 * bias
        right_nosewing[-3][0] = right_nosewing[-3][0] + bias
        right_nosewing[1][0] = right_nosewing[1][0] - bias
        right_nosewing[2][0] = right_nosewing[2][0] - bias
        right_nosewing = self.get_soft_line(right_nosewing)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [right_nosewing], (1, 1, 1))
        mask = img * mask
        return mask, [right_nosewing]

    def eyelash(self, img):
        left_eye_center_y = (self.keypoint_all[450][1] - self.keypoint_all[257][1]) / 2 + \
                            self.keypoint_all[257][1]
        right_eye_center_y = (self.keypoint_all[230][1] - self.keypoint_all[27][1]) / 2 + \
                             self.keypoint_all[27][1]

        left_eye_top = self.keypoint_all[[414, 286, 258, 257, 259]]
        left_eye_top[[0], 1] = left_eye_top[[0], 1] + 0.1 * (left_eye_center_y - left_eye_top[[0], 1])
        left_eye_top[1:, 1] = left_eye_top[1:, 1] + 0.3 * (left_eye_center_y - left_eye_top[1:, 1])
        left_eye_top[[2, 3], 1] = left_eye_top[[2, 3], 1] + 0.2 * (left_eye_center_y - left_eye_top[[2, 3], 1])
        left_eye_top = np.vstack((left_eye_top, self.keypoint_all[[359]]))
        left_eye_bottom = self.keypoint_all[[261, 449, 450, 451, 453]]
        left_eye_bottom[:, 1] = left_eye_bottom[:, 1] - 0 * (left_eye_bottom[:, 1] - left_eye_center_y)
        left_eye = np.vstack((left_eye_top, left_eye_bottom))

        right_eye_top = self.keypoint_all[[29, 27, 28, 56, 190]]
        right_eye_top[[-1], 1] = right_eye_top[[-1], 1] + 0.1 * (right_eye_center_y - right_eye_top[[-1], 1])
        right_eye_top[:-1, 1] = right_eye_top[:-1, 1] + 0.3 * (right_eye_center_y - right_eye_top[:-1, 1])
        right_eye_top[[1, 2], 1] = right_eye_top[[1, 2], 1] + 0.2 * (right_eye_center_y - right_eye_top[[1, 2], 1])
        right_eye_top = np.vstack((self.keypoint_all[[130]], right_eye_top))
        right_eye_bottom = self.keypoint_all[[233, 231, 230, 229, 31]]
        right_eye_bottom[:, 1] = right_eye_bottom[:, 1] - 0 * (right_eye_bottom[:, 1] - right_eye_center_y)
        right_eye = np.vstack((right_eye_top, right_eye_bottom))

        mask = np.zeros(img.shape, dtype=np.uint8)
        left_eye = self.get_soft_line(left_eye)
        right_eye = self.get_soft_line(right_eye)
        cv2.fillPoly(mask, [left_eye, right_eye], (1, 1, 1))
        mask = img * mask
        return mask, [left_eye, right_eye]

    def left_eyelash(self, img):
        left_eye_center_y = (self.keypoint_all[450][1] - self.keypoint_all[257][1]) / 2 + \
                            self.keypoint_all[257][1]

        left_eye_top = self.keypoint_all[[414, 286, 258, 257, 259]]
        left_eye_top[[0], 1] = left_eye_top[[0], 1] + 0.1 * (left_eye_center_y - left_eye_top[[0], 1])
        left_eye_top[1:, 1] = left_eye_top[1:, 1] + 0.3 * (left_eye_center_y - left_eye_top[1:, 1])
        left_eye_top[[2, 3], 1] = left_eye_top[[2, 3], 1] + 0.2 * (left_eye_center_y - left_eye_top[[2, 3], 1])
        left_eye_top = np.vstack((left_eye_top, self.keypoint_all[[359]]))
        left_eye_bottom = self.keypoint_all[[261, 449, 450, 451, 453]]
        left_eye_bottom[:, 1] = left_eye_bottom[:, 1] - 0 * (left_eye_bottom[:, 1] - left_eye_center_y)
        left_eye = np.vstack((left_eye_top, left_eye_bottom))

        mask = np.zeros(img.shape, dtype=np.uint8)
        left_eye = self.get_soft_line(left_eye)
        cv2.fillPoly(mask, [left_eye], (1, 1, 1))
        mask = img * mask
        return mask, [left_eye]

    def right_eyelash(self, img):
        right_eye_center_y = (self.keypoint_all[230][1] - self.keypoint_all[27][1]) / 2 + \
                             self.keypoint_all[27][1]
        right_eye_top = self.keypoint_all[[29, 27, 28, 56, 190]]
        right_eye_top[[-1], 1] = right_eye_top[[-1], 1] + 0.1 * (right_eye_center_y - right_eye_top[[-1], 1])
        right_eye_top[:-1, 1] = right_eye_top[:-1, 1] + 0.3 * (right_eye_center_y - right_eye_top[:-1, 1])
        right_eye_top[[1, 2], 1] = right_eye_top[[1, 2], 1] + 0.2 * (right_eye_center_y - right_eye_top[[1, 2], 1])
        right_eye_top = np.vstack((self.keypoint_all[[130]], right_eye_top))
        right_eye_bottom = self.keypoint_all[[233, 231, 230, 229, 31]]
        right_eye_bottom[:, 1] = right_eye_bottom[:, 1] - 0 * (right_eye_bottom[:, 1] - right_eye_center_y)
        right_eye = np.vstack((right_eye_top, right_eye_bottom))

        mask = np.zeros(img.shape, dtype=np.uint8)
        right_eye = self.get_soft_line(right_eye)
        cv2.fillPoly(mask, [right_eye], (1, 1, 1))
        mask = img * mask
        return mask, [right_eye]


        # 眉毛

    def eyebrow(self, img):
        left_eb_h = self.keypoint_all[285][1] - self.keypoint_all[336][1]
        left_eb_top = self.keypoint_all[[336, 334, 300]]
        left_eb_top[:, 1] = left_eb_top[:, 1] - 0.6 * left_eb_h
        left_eb_bottom = self.keypoint_all[[276, 282, 285]]
        left_eb_bottom[:, 1] = left_eb_bottom[:, 1] + 0.6 * left_eb_h
        left_eb = np.vstack((left_eb_top, left_eb_bottom))
        left_eb[[0, 5], 0] = left_eb[[0, 5], 0] - 30
        left_eb[[2, 3], 0] = left_eb[[2, 3], 0] + 50

        right_eb_h = self.keypoint_all[55][1] - self.keypoint_all[107][1]
        right_eb_top = self.keypoint_all[[70, 105, 107]]
        right_eb_top[:, 1] = right_eb_top[:, 1] - 0.6 * right_eb_h
        right_eb_bottom = self.keypoint_all[[55, 52, 46]]
        right_eb_bottom[:, 1] = right_eb_bottom[:, 1] + 0.6 * right_eb_h
        right_eb = np.vstack((right_eb_top, right_eb_bottom))
        right_eb[[0, 5], 0] = right_eb[[0, 5], 0] - 50
        right_eb[[2, 3], 0] = right_eb[[2, 3], 0] + 30

        left_eb = self.get_soft_line(left_eb)
        right_eb = self.get_soft_line(right_eb)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [left_eb, right_eb], (1, 1, 1))
        mask = img * mask
        return mask, [left_eb, right_eb]

        # 嘴唇

    def lips(self, img):
        lips_top = self.keypoint_all[[185, 39, 37, 267, 269, 409]]
        lips_top[:, 1] = lips_top[:, 1] - 80
        lips_61 = [self.keypoint_all[61][0] - 80, self.keypoint_all[61][1]]
        lips_291 = [self.keypoint_all[291][0] + 80, self.keypoint_all[291][1]]

        lips_top = np.vstack((lips_61, lips_top, lips_291))
        lips_bottom = self.keypoint_all[[321, 405, 314, 17, 84, 181, 91]]
        lips_bottom[:, 1] = lips_bottom[:, 1] + 50

        lips = np.vstack((lips_top, lips_bottom))
        lips = self.get_soft_line(lips)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [lips], (1, 1, 1))
        mask = img * mask
        return mask, [lips]

        # 下巴下面托盘

    def jaw_bottom(self, img):
        jaw_bottom = self.keypoint_all[[149, 32, 201, 200, 421, 262, 378, 400, 377, 152, 148, 176]]
        mask = np.zeros(img.shape, dtype=np.uint8)
        jaw_bottom = self.get_soft_line(jaw_bottom)
        cv2.fillPoly(mask, [jaw_bottom], (1, 1, 1))
        mask = img * mask
        return mask, [jaw_bottom]

        # 鼻孔

    def nostril(self, img):
        nostril_top = self.keypoint_all[[79, 44, 19, 274, 309]]
        nostril_top[:, 1] = nostril_top[:, 1] - 30
        nostril_bottom = self.keypoint_all[[460, 2, 98]]
        nostril = np.vstack((nostril_top, nostril_bottom))

        mask = np.zeros(img.shape, dtype=np.uint8)
        nostril = self.get_soft_line(nostril)
        cv2.fillPoly(mask, [nostril], (1, 1, 1))
        mask = img * mask
        return mask, [nostril]

    def nostril_shadow(self, img):
        nostril_top = self.keypoint_all[[79, 44, 19, 274, 309]]
        nostril_top[2][1] = nostril_top[2][1] - 30
        nose_shadow_left2 = self.keypoint_all[[358, 429]]
        nose_shadow_left1 = self.keypoint_all[[429, 358, 327]]
        nose_shadow_left1[:, 0] = nose_shadow_left1[:, 0] + 45
        nose_shadow_left2[:, 0] = nose_shadow_left2[:, 0] - 130
        nostril_top[:, 1] = nostril_top[:, 1] - 30

        nostril_bottom = self.keypoint_all[[2]]
        nose_shadow_right2 = self.keypoint_all[[98, 129, 209]]
        nose_shadow_right1 = self.keypoint_all[[209, 129]]
        nose_shadow_right1[:, 0] = nose_shadow_right1[:, 0] + 130
        nose_shadow_right2[:, 0] = nose_shadow_right2[:, 0] - 45

        nostril_shadow = np.vstack(
            (nostril_top, nose_shadow_left2, nose_shadow_left1, nostril_bottom, nose_shadow_right2, nose_shadow_right1))

        mask = np.zeros(img.shape, dtype=np.uint8)
        nostril_shadow = self.get_soft_line(nostril_shadow)
        cv2.fillPoly(mask, [nostril_shadow], (1, 1, 1))
        mask = img * mask
        return mask, [nostril_shadow]

    def between_eyebrows(self, img):
        bias = 81
        betweeen_eyebrows = self.keypoint_all[[151, 337, 336, 285, 168, 55, 107, 108]]
        betweeen_eyebrows[2][0] = betweeen_eyebrows[2][0] - 1 * bias
        betweeen_eyebrows[3][0] = betweeen_eyebrows[3][0] - 1 * bias
        betweeen_eyebrows[-2][0] = betweeen_eyebrows[-2][0] + 1 * bias
        betweeen_eyebrows[-3][0] = betweeen_eyebrows[-3][0] + 1 * bias
        betweeen_eyebrows = self.get_soft_line(betweeen_eyebrows)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [betweeen_eyebrows], (1, 1, 1))
        mask = img * mask
        return mask, [betweeen_eyebrows]


    def all_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)


        other = self.keypoint_all[[300,345,433,364,395,
                         262,200,32,170,135,213,116,70]]
        keypoint_new = np.vstack((keypoint_new,other))
        mask = np.zeros(img.shape, dtype=np.uint8)
        keypoint_new = self.get_soft_line(keypoint_new)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask[mask_gray < 50] = 0
        return mask,[keypoint_new]

    def acne_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)


        other = self.keypoint_all[[300,444,442,413,453,449,

                        345,433,364,395,
                         262,200,32,170,135,213,116,
                         229,233,189,222,224,70

                                   ]]
        keypoint_new = np.vstack((keypoint_new,other))
        mask = np.zeros(img.shape, dtype=np.uint8)
        keypoint_new = self.get_soft_line(keypoint_new)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask[mask_gray < 50] = 0
        return mask,[keypoint_new]

    def porphyrin_face(self, img):
        forehead_h = self.keypoint_all[151][1] - self.keypoint_all[10][1]
        # forehead_w = keypoint[333][0] - keypoint[104][0]
        forehead_center_x = (self.keypoint_all[10][0] + self.keypoint_all[151][0]) / 2
        # forehead_center_y = (keypoint[10][1] + keypoint[151][1])/2
        # 顺时针
        keypoint_new = []
        # 左侧
        keypoint_103 = [self.keypoint_all[103][0] + 0.01 * (forehead_center_x - self.keypoint_all[103][0]),
                        self.keypoint_all[103][1]]
        keypoint_new.append(keypoint_103)

        # 上方
        keypoint_top = [67, 109, 10, 338, 297]
        top_y_bias = 1
        top_x_bias = 0.05
        for point in keypoint_top:
            if point == 10:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.06) * forehead_h]
            # elif point ==103 or 332:
            #     k_p = [self.keypoint_all[point][0] + top_x_bias*1.1 * (forehead_center_x - self.keypoint_all[point][0]),
            #            self.keypoint_all[point][1] - (top_y_bias*0.8) * forehead_h]
            elif point in [338, 109]:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 1.05) * forehead_h]
            else:
                k_p = [self.keypoint_all[point][0] + top_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                       self.keypoint_all[point][1] - (top_y_bias * 0.9) * forehead_h]
            keypoint_new.append(k_p)

        # 右侧
        keypoint_332 = [self.keypoint_all[332][0] + 0.01 * (forehead_center_x - self.keypoint_all[332][0]),
                        self.keypoint_all[332][1]]
        keypoint_new.append(keypoint_332)

        # 下方
        keypoint_bottom = [298, 333, 299, 69, 104, 68]
        bottom_y_bias = 0.4
        bottom_x_bias = 0.05
        for point in keypoint_bottom:
            k_p = [self.keypoint_all[point][0] + bottom_x_bias * (forehead_center_x - self.keypoint_all[point][0]),
                   self.keypoint_all[point][1] + bottom_y_bias * forehead_h]
            keypoint_new.append(k_p)

        other = self.keypoint_all[[336,465,451,449,346,352,
                                   416,430,422,410,393,167,186,202,
                                   210,192,123,117,229,231,245,
                                   107]]
        other[[0],0] = other[[0],0]-40
        other[[4],1] = other[[4],1]-40
        other[[-1],0] = other[[-1],0]+40
        other[[-5],1] = other[[-5],1]-40
        other[[0,-1],1] = other[[0,-1],1]+40
        keypoint_new = np.array(keypoint_new)
        keypoint_new = np.insert(keypoint_new, -3, other, axis=0)
        # 单个闭合图形保持整数
        keypoint_new = np.array(keypoint_new).astype(np.int32)

        keypoint_new = self.get_soft_line(keypoint_new)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [keypoint_new], (1, 1, 1))
        mask = img * mask
        return mask, [keypoint_new]

    def eye_mask(self, img):
        key_list = self.keypoint_all[[446, 261, 448, 449, 450, 451, 452, 6,
                                      232, 231, 230, 229, 228, 31, 226,
                                      113, 225, 222, 442, 445, 342]]
        mask = np.zeros(img.shape, dtype=np.uint8)
        key_list = self.get_soft_line(key_list)
        cv2.fillPoly(mask, [key_list], (1, 1, 1))
        mask = img * mask
        return mask, [key_list]


def adjust(keypoints):
    keypoint_top = [54,103,67,109,10,338,297,332,284]
    for point in keypoint_top:
        keypoints[point][1] -= 140
    keypoint_top = [68,104,69,108,151,337,299,333,298]
    for point in keypoint_top:
        keypoints[point][1] -= 80
    return keypoints


# def draw(keypoints,img):
#     for xy in keypoints:
#         x,y = xy
#         # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.circle(img, (x, y), 8, (50, 255, 50), -1)
#     return img


from face_detector import FaceSeg
def face_seg(img):
    a = time.time()
    seg = FaceSeg("face_seg.onnx")
    face_mask = seg.inference(img)
    print("Face seg : ", round(time.time() - a, 2))
    return face_mask

def face_crop(img, parts_list):
    img = usm(img)
    img_kp_copy, img_copy, right_left = img.copy(), img.copy(), img.copy()
    
    keypoint = get_face_mesh(img_kp_copy)
    if len(keypoint) == 0:
        return None, None, None, None, None, []
    cut_img = CutImg(keypoint)
    
    mask_new, xyxy, keypoint_parts = cut_img.get_parts(img, parts_list)
    
    rect_img = img_copy[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
    parts_mask = np.zeros(mask_new.shape,dtype=np.uint8)
    for keypoint_part in keypoint_parts:
        cv2.fillPoly(parts_mask, [np.array(keypoint_part[0], dtype=np.int32)], (1, 1, 1))


    right_points, left_points = cut_img.get_area(256)
    # right, left = right_left[right_points[0][1]:right_points[1][1], right_points[0][0]:right_points[1][0], :], right_left[left_points[0][1]:left_points[1][1], left_points[0][0]:left_points[1][0], :]
    
    return rect_img, parts_mask, xyxy, right_points, left_points, keypoint

if __name__ == "__main__":
    dir_path = r'input'
    out_path = "output/"
    os.makedirs(out_path,exist_ok=True)
    for file in os.listdir(dir_path):
            name = file.split(".")[0]
            path = os.path.join(dir_path,file)
            img = cv2.imread(path)
            if img is None:
                continue
            img = usm(img)
            img_cp, ori = img.copy(), img.copy()
            
            # keypoint是mp自带检测出的人脸关键点
            keypoint = get_face_mesh(img_cp)
            # keypoint_new = adjust(keypoint)
            # img_new = img.copy()
            # img_res = draw(keypoint_new,img_new)
            # cv2.imwrite(out_path+name + "mesh_new.jpg", img_res)


            if len(keypoint) ==0:
                continue
            

            cut_img = CutImg(keypoint)
            a = time.time()

            # face_mask是onnx的output，为原始大小
            face_mask = face_seg(img)

            # 获取通过mask分割出的人脸
            img[face_mask == 0] = [0, 0, 0]

            mask_new, xyxy, keypoint_parts = cut_img.get_parts(img,['left_pore_area','right_pore_area'])

            # 局部矩形图像rect_img
            rect_img = ori[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
            cv2.imwrite(out_path+name + "parts.jpg", rect_img)

            # parts_mask是物体在局部图像上的掩膜
            parts_mask = np.zeros(mask_new.shape,dtype=np.uint8)
            for keypoint_part in keypoint_parts:
                cv2.fillPoly(parts_mask, [np.array(keypoint_part[0], dtype=np.int32)], (1, 1, 1))
            # break
            
            # parts_img表现=mask_new
            parts_img = parts_mask * rect_img
            # print(mask_new.shape)
            # break
            cv2.imwrite(out_path+name+"cut.jpg", parts_img)
            # img_cp1 = img.copy()
            # for kp_part in keypoint_parts:
            #     img_cp1 = cv2.polylines(img_cp1, kp_part, True, (85, 186, 255), 10)
            #     for circle in kp_part:
            #         for kp in circle:
            #             cv2.circle(img_cp, kp, 10, (0, 0, 255), 2)
            # cv2.imwrite(out_path+name+"line.jpg",img_cp1)
            cv2.imwrite(out_path+name + "mesh.jpg", img_cp)
            #
            #
            #
