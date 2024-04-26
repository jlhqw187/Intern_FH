"""
绘制下颌线
input: 人脸图像
output: 三个点[起始点， 中位点， 终点]， 下颌线。
"""
import cv2
import numpy as np
import mediapipe.python.solutions as mp
import onnxruntime as rt

class FaceDetector:
    def __init__(self):
        self.mpface_mesh = mp.face_mesh
        self.face_mesh = self.mpface_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        # self.mpdraw = mp.drawing_utils
        # self.draw_spec = self.mpdraw.DrawingSpec(thickness=5, circle_radius=5, color=(50, 255, 50))

    def get_face_mesh(self, img0):
        img = img0.copy()
        h,w,_ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_img = self.face_mesh.process(img_rgb)
        keypoint = []
        flag = 0
        face_rate = 0.3
        if result_img.multi_face_landmarks:
            face_lms = result_img.multi_face_landmarks[0]
            # self.mpdraw.draw_landmarks(img, face_lms, self.mpface_mesh.FACEMESH_CONTOURS,
            #                            self.draw_spec, self.draw_spec)
            for id, lm in enumerate(face_lms.landmark):
                h, w, c = img.shape
                y, x = int(lm.y * h), int(lm.x * w)
                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.circle(img,(x,y),2,(50,255,50),-1)                
                keypoint.append([x, y])
            keypoint = np.array(keypoint)
            # face_h = max(keypoint[:, 1]) - min(keypoint[:, 1])
            # face_w = max(keypoint[:, 0]) - min(keypoint[:, 0])
            # face_proportion = (face_w * face_h) / (w * h)
            # print('face = {}'.format(face_proportion))
            # flag = 1 if face_proportion > face_rate else 0
            
        return keypoint



import os


def draw_jaw_line(img):
    img_mesh, ori, img_flood, img_result = img.copy(), img.copy(), img.copy(), img.copy()

    keypoint = FaceDetector().get_face_mesh(img_mesh)
    skin_point = keypoint[199]
    jaw_line_point = keypoint[164]
    bottom_point = keypoint[152]

    h, w = img.shape[:2]
    flood_mask = np.zeros((h+2, w+2, 1), np.uint8)
    seed_point = skin_point
    low_diff = (150, ) * 3
    up_diff = (150, ) * 3 
    cv2.floodFill(img_flood, flood_mask, seed_point, (255, 255, 255), low_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
    flood_mask = flood_mask[1:h+1, 1:w+1]
    
    flood_mask[0:jaw_line_point[1]] = 0
    flood_mask = (flood_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 50000:
            x_coordinates = contour[:, 0, 0]
            max_x_index = np.argmax(x_coordinates)
            min_x_index = np.argmin(x_coordinates)
            min_x_index, max_x_index = min(min_x_index, max_x_index), max(min_x_index, max_x_index)
            contour = contour[min_x_index : max_x_index, :, :]
            
            # contour_x_distance = abs(contour[:, :, 0] - bottom_point[0])
            # contour_center_point = contour[np.where(contour_x_distance == min(contour_x_distance))[0], :, :][0][0]
            # cv2.circle(img_result, contour_center_point, 20, (255, 0, 0), -1)

            
            x_coordinates = contour[:, 0, 0]
            y_coordinates = contour[:, 0, 1]

            # indices = []
            # for i in range(1, len(x_coordinates), 1):
            #     if x_coordinates[i] - x_coordinates[i-1] < 0:
            #         indices.append(i)
            # for i in range(1, len(y_coordinates), 1):
            #     if y_coordinates[i] - y_coordinates[i-1] > 0:
            #         indices.append(i)
            # mask = np.ones(len(contour), dtype=bool)
            # mask[indices] = False
            # contour = contour[mask, :, :]

            coefficients = np.polyfit(x_coordinates, y_coordinates, 30)
            polynomial = np.poly1d(coefficients)
            x_smooth = np.linspace(x_coordinates.min(), x_coordinates.max(), 2000)
        
            # x_smooth = x_smooth[:, np.newaxis]
            y_smooth = polynomial(x_smooth)
            fitting = np.stack((x_smooth, y_smooth), axis=-1)
            fitting = fitting[:, np.newaxis, :]
            fitting = fitting.astype(np.int32)

            fitting_x_distance = abs(fitting[:, :, 0] - bottom_point[0])
            fitting_center_point = fitting[np.where(fitting_x_distance == min(fitting_x_distance))[0], :, :][0][0]
            
            left_jaw_point, right_jaw_point = fitting[0][0], fitting[-1][0]
            # cv2.circle(img_result, fitting_center_point, 10, (0, 255, 0), -1)
            # cv2.circle(img_result, left_jaw_point, 10, (0, 255, 0), -1)
            # cv2.circle(img_result, right_jaw_point, 10, (0, 255, 0), -1)
            # cv2.polylines(img_result, [fitting], False, (0, 0, 255), 5)

    return [left_jaw_point, fitting_center_point, right_jaw_point], fitting

if __name__ == "__main__":
    input_path = r"F:\fh\draw_jaw_line\center/"
    file_list = os.listdir(input_path)
    for file in file_list:
        img = cv2.imread(input_path + file)
        points, jaw_line = draw_jaw_line(img)
        print(len(points), len(jaw_line))
