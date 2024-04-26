import cv2
import numpy as np

def usm(img, w=0.3):
    # 自动确定高斯核的大小，标准差为5，越大越模糊
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    a = 1 / (1 - w)
    b = -w / (1 - w)
    usm_out = cv2.addWeighted(img, a, blur_img, b, 0)
    return usm_out

def uv_correct(img, h_change=10, s_change=3, v_change=0, contrast=1.0, brightness=0):
    blank = np.zeros(img.shape, img.dtype)
    img = cv2.addWeighted(img, contrast, blank, 1 - contrast, brightness)

    uv_img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_change = int(h_change / 360 * 180) # ps中h范围为-180~180
    s_change = int(s_change / 200 * 255) # ps中s范围为-100~100
    h = uv_img_hsv[..., 0].astype(np.int16)
    s = uv_img_hsv[..., 1].astype(np.int16)
    v = uv_img_hsv[..., 2].astype(np.int16)
    h = np.clip(h + h_change, 0, 180).astype(np.uint8)
    s = np.clip(s + s_change, 0, 255).astype(np.uint8)
    v = np.clip(v + v_change, 0, 255).astype(np.uint8)
    uv_img_hsv = cv2.merge([h, s, v])
    uv_img = cv2.cvtColor(uv_img_hsv, cv2.COLOR_HSV2BGR)
    return uv_img

def lep_preprocess(image, sigma_s=150, sigma_r=0.1):
    """base为光强层，detail为细节层"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    
    ## 距离转换滤波器
    base_layer = cv2.ximgproc.dtFilter(image, image, sigma_s, sigma_r)
    detail_layer = image - base_layer
    base_layer = np.uint8(base_layer * 255)

    detail_layer_unstretch = np.uint8(detail_layer * 255)
    # cv2.imwrite("detail_layer_unstretch.jpg", detail_layer)
    
    detail_layer = cv2.normalize(detail_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    detail_layer = np.uint8(detail_layer)
    

    return base_layer, detail_layer

def DoG(image, sigma1=3, sigma2=3*1.6):
    """高斯差分"""
    sigma1 = 3 
    sigma2 = 27
    gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    # 替换掉了abs，现在呈现双峰的态势
    dog = gaussian1 - gaussian2
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog

def pg_preprocess(image):
    """取亮度->CLAHE->DoG"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_y = image[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image_y)
    cv2.imwrite("clahed.jpg", clahe_image)

    dog = DoG(clahe_image)
    cv2.imwrite("dog.jpg", dog)

    return dog

def detect_preprocess_original(rect_img):
    """原始做法"""
    rect_img = rect_img.copy()
    
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    gray = usm(gray, w=0.5)
    
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      121, 5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

def detect_preprocess_pg(rect_img):
    """宝洁的pipeline与原始有较大不同:
    直接用阈值划分-》闭运算而不是开运算-》findcontours的时候要查找内部轮廓(否则会只检测到最外侧的轮廓)"""
    rect_img = rect_img.copy()
    
    gray = pg_preprocess(rect_img)
    
    thresh_value = 120
    ret, threshold = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

def detect_preprocess_lep(rect_img):
    """lep的pipeline和原始的基本一致"""
    rect_img = rect_img.copy()
    
    _, gray = lep_preprocess(rect_img)
    
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,121,5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)