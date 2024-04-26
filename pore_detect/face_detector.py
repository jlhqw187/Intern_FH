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

    def get_face_mesh(self, img0, direction='left'):
        img = img0.copy()
        h,w,_ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_img = self.face_mesh.process(img_rgb)
        keypoint = []
        flag = 0
        # todo(人脸占比后续可用宽度w替代)
        face_rate = 0.3
        if result_img.multi_face_landmarks:
            face_lms = result_img.multi_face_landmarks[0]
            # self.mpdraw.draw_landmarks(img, face_lms, self.mpface_mesh.FACEMESH_CONTOURS,
            #                            self.draw_spec, self.draw_spec)
            for id, lm in enumerate(face_lms.landmark):
                h, w, c = img.shape
                y, x = int(lm.y * h), int(lm.x * w)
                # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                keypoint.append([x, y])
            keypoint = np.array(keypoint)
            face_h = max(keypoint[:, 1]) - min(keypoint[:, 1])
            face_w = max(keypoint[:, 0]) - min(keypoint[:, 0])
            face_proportion = (face_w * face_h) / (w * h)
            print('face = {}'.format(face_proportion))
            flag = 1 if face_proportion > face_rate else 0
            
        return keypoint, img, flag

    # def save_img_json(self, point_position, img, output_path):
    #     cv2.imwrite(output_path + 'keypoint.jpg', img)
    #     with open(output_path + 'keypoint.json', 'w') as f:
    #         f.write(json.dumps({'keypoint': point_position.tolist()}, indent=4))

class FaceSeg():
    def __init__(self,seg_path):
        self.sess = rt.InferenceSession(seg_path)
        self.face_id = [1,2,3,4,5,6,10,11,12,13]

    def preprocess(self,img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(512,512))
        img = np.float32(img / 255.0)
        img = (img - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img,axis=0)
        return img

    def get_face_mask(self,pred):
        face_mask = np.zeros(pred.shape,np.uint8)
        for idx in self.face_id:
            face_mask[pred == idx] = 1
        return face_mask

    def inference(self,raw_img):
        h, w, _ = raw_img.shape
        img = self.preprocess(raw_img)
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        pred = self.sess.run([label_name], {input_name: img})[0][0]
        masks_pred = np.array(pred.argmax(0))
        masks_pred = cv2.resize(masks_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        masks_pred = masks_pred.astype(np.uint8)
        face_mask = self.get_face_mask(masks_pred)
        return face_mask
