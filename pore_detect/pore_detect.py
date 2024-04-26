"""
截止425的毛孔检测算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 锐化操作：usm
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

def is_region_of_interest(target, query):
    """计算query的两个点组成的矩形区域是否在target内"""
    target_upper_left, target_lower_right = target
    query_upper_left, query_lower_right = query
    return (target_upper_left[0] < query_upper_left[0] and target_upper_left[1] < query_upper_left[1]) and (target_lower_right[0] > query_lower_right[0] and target_lower_right[1] > query_lower_right[1]) 



def find_rect(cnt):
    """获取contour的外接矩形 w * h 与 2w * 2h"""
    # contour的外接矩形的左上角和右下角坐标
    upper_left = (cnt[:, :, 1].min(), cnt[:, :, 0].min())
    lower_right = (cnt[:, :, 1].max(), cnt[:, :, 0].max())
    rect_points = (upper_left, lower_right)

    # contour的两倍长宽外接矩形，即带背景的外接矩形的左上角和右下角坐标
    upper_left_bg = ((3*upper_left[0] - lower_right[0])//2, (3*upper_left[1] - lower_right[1])//2)
    lower_right_bg = ((3*lower_right[0] - upper_left[0])//2, (3*lower_right[1] - upper_left[1])//2)
    rect_points_bg = (upper_left_bg, lower_right_bg)
    return rect_points, rect_points_bg

def is_stain_BGR(img, points):
    """使用BGR通道判断 检测出的毛孔是否是色斑，具体是通过对比背景信息和中心信息来确定"""
    n = 7

    center = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
    core_points = ((center[0] - n // 2, center[1] - n // 2),  (center[0] + n // 2, center[1] + n // 2))
    core = img[core_points[0][0] : core_points[1][0]+1, core_points[0][1] : core_points[1][1]+1]
    # core_hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    core_mean = core.mean()
    
    m = 4
    outer_points = ((points[0][0] - m//2, points[0][1] - m//2), (points[1][0] + m//2, points[1][1] + m//2))
    img_copy = img.copy() 
    img_copy[points[0][0] : points[1][0]+1, points[0][1] : points[1][1]+1] = 0
    outer = img_copy[outer_points[0][0]: outer_points[1][0]+1, outer_points[0][1]: outer_points[1][1]+1]
    # outer = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)

    mask = np.ma.masked_where(outer == 0, outer)
    outer_mean = np.mean(mask)

    thresh = 25
    if(outer_mean - core_mean > thresh):
        return True
    else:
        return False

def is_stain_HSV(img, points):
    """通过HSV通道来判断"""
    img_copy = img.copy()
    # outer_img_copy = img.copy()
    h = (points[1][0] - points[0][0]) // 2
    w = (points[1][1] - points[0][1]) // 2
    n = [h // 2, w //2]
        
    # center = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
    # core_points = ((center[0] - n // 2, center[1] - n // 2),  (center[0] + n // 2, center[1] + n // 2))
    # core_points = 
    core = img_copy[h*3//4 : (h*5//4)+1, w*3//4 : (w*5//4)+1, ]
    # core_hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    # core_mean = core.mean()
    core_h_mean = core[:, :, 0].mean()
    core_s_mean = core[:, :, 1].mean()
    core_v_mean = core[:, :, 2].mean()
    # cv2.rectangle(core_img_copy, core_points[0][::-1], core_points[1][::-1], (0, 0, 255), 1)
    

    m = 4
    # outer_points = points
    # img_copy[core_points[0][0] : core_points[1][0]+1, core_points[0][1] : core_points[1][1]+1] = 0
    img_copy[h*3//4 : (h*5//4)+1, w*3//4 : (w*5//4)+1, ] = 0
    # outer_points = ((points[0][0] - m//2, points[0][1] - m//2), (points[1][0] + m//2, points[1][1] + m//2))
    # outer_img_copy[points[0][0] : points[1][0]+1, points[0][1] : points[1][1]+1] = 0
    # outer = outer_img_copy[outer_points[0][0]: outer_points[1][0]+1, outer_points[0][1]: outer_points[1][1]+1]
    
    # if outer == None:
    #     return False
    # outer_hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)

    num_of_outer = (img_copy.shape[0] * img_copy.shape[1]) * 3 / 4
    # outer_value = outer_hsv[0:m//2, :, :] + 
    # 多加一重验证，如果outer里已经是bg黑色的部分，则剔除黑色bg的部分再计算均值
    # ==0的地方 = np.nan，再用np.nanmean()来求
    # num_of_outer = 2 * (m//2) * (points[1][0] - points[0][0] + 1) + 2 * (m//2) * (points[1][1] - points[0][1] + 1) + 4 * (m//2) * (m//2)
    # outer_h_mean = outer_hsv[:, :, 0].sum() / num_of_outer
    # outer_s_mean = outer_hsv[:, :, 1].sum() / num_of_outer
    # outer_v_mean = outer_hsv[:, :, 2].sum() / num_of_outer

    # 记住np.nan的位置，和非nan的位置，非nan的位置转换成HSV，在非nan的位置求和后/非nan的总数
    # outer_img_copy = outer_img_copy.astype(float)
    # outer_img_copy[points[0][0] : points[1][0]+1, points[0][1] : points[1][1]+1, :] = np.nan
    # outer_img_copy = np.where(outer_img_copy == 0, np.nan, outer_img_copy)
    # # outer_img_copy = np.where(outer_img_copy == np.nan, 0, outer_img_copy)
    # outer_img_copy = outer_img_copy.astype(int)
    # # np.where(outer)

    # outer = outer_img_copy[outer_points[0][0]: outer_points[1][0]+1, outer_points[0][1]: outer_points[1][1]+1]
    # outer_hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)

    img_copy = np.where(img_copy == 0, np.nan, img_copy)
    new_h_mean = np.nanmean(img_copy[:, :, 0])
    new_s_mean = np.nanmean(img_copy[:, :, 1])
    new_v_mean = np.nanmean(img_copy[:, :, 2])


    if (core_s_mean - new_s_mean) > 10 and  (new_v_mean - core_v_mean) > 10:
        return True
    else:
        return False


def is_stain_UV(img_uv, points):
    """通过UV通道来判断"""
    img_uv = img_uv.copy()

    h = (points[1][0] - points[0][0]) // 2
    w = (points[1][1] - points[0][1]) // 2
    n = [h // 2, w //2]


    core = img_uv[h*3//4 : (h*5//4)+1, w*3//4 : (w*5//4)+1, ]
    core_h_mean = core[:, :, 0].mean()
    core_s_mean = core[:, :, 1].mean()
    core_v_mean = core[:, :, 2].mean()

    img_uv[h*3//4 : (h*5//4)+1, w*3//4 : (w*5//4)+1, ] = 0
    img_uv = np.where(img_uv == 0, np.nan, img_uv)
    new_h_mean = np.nanmean(img_uv[:, :, 0])
    new_s_mean = np.nanmean(img_uv[:, :, 1])
    new_v_mean = np.nanmean(img_uv[:, :, 2])

    if (new_v_mean - core_v_mean) > 15 and (core_s_mean - new_s_mean) > 5:
        return True
    else:
        return False



def make_seg_mask(parts_img, contour_list):
    """在包含part的img获取contour的mask"""
    seg_mask = np.zeros_like(parts_img)
    skin_mask = parts_img.copy()

    for contour in contour_list:
        cv2.fillPoly(skin_mask, pts =[contour], color=(0, 0, 0))
        cv2.fillPoly(seg_mask, pts =[contour], color=(255, 255, 255))

    skin_mask = np.where(skin_mask == 0, skin_mask, 1)
    
    color_array = np.full(parts_img.shape, (0, 0, 0), dtype=np.uint8)
    seg_mask = np.where(skin_mask, color_array, seg_mask)
    return seg_mask

def preprocess(image, sigma_s=150, sigma_r=0.1):
    """见preprocess文件夹"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    
    base_layer = cv2.ximgproc.dtFilter(image, image, sigma_s, sigma_r)
    detail_layer = image - base_layer
    base_layer = np.uint8(base_layer * 255)

    # detail_layer_unstretch = np.uint8(detail_layer * 255)
    # cv2.imwrite("detail_layer_unstretch.jpg", detail_layer)
    
    detail_layer = cv2.normalize(detail_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    detail_layer = np.uint8(detail_layer)
    

    return base_layer, detail_layer

def DoG(image, sigma1=3, sigma2=3*1.6):
    """见preprocess文件夹"""
    sigma1 = 3 
    sigma2 = 27
    gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    # dog = cv2.absdiff(gaussian1, gaussian2)
    gaussian1 = gaussian1.astype(np.float32)
    gaussian2 = gaussian2.astype(np.float32)
    dog = gaussian1 - gaussian2
     # dog = (dog - mean) / std
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    dog = dog.astype(np.uint8)
    return dog

def pg_preprocess(image):
    """见preprocess文件夹"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_y = image[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image_y)
    cv2.imwrite("clahed.jpg", clahe_image)

    dog = DoG(clahe_image)
    cv2.imwrite("dog.jpg", dog)

    return dog
    


def detect(rect_img, max_area, parts_mask, rect_img_uv):
    """
    rect_img是包含part，无黑边的矩形图像， parts_mask为部位的掩码，rect_img_uv是为了方便后面结合uv一起做实际现在并没用。
    注释的部分主要是一开始用is_stain_BGR等等做实验时候用到的，这里用不上，但是保留注释。
    """
    output_folder = r"./temp/"
    img = rect_img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_uv = rect_img_uv.copy()
    rect_img_uv_hsv = cv2.cvtColor(img_uv, cv2.COLOR_BGR2HSV)

    height, width = img.shape[:2]
    rect_img = rect_img.copy()

    # adp_threshold的值如何来的？ 超参数 or 聚类得出
    ori1, ori2, ori3, ori4 = rect_img.copy(), rect_img.copy(), rect_img.copy(), rect_img.copy()
    cv2.imwrite(output_folder + "ori.jpg", ori1)
    adp_threshold = [121, 5]


    # 先转灰度图，再做锐化处理
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_folder + "gray.jpg", gray)

    # why 0.5？
    gray = usm(gray, w=0.5)
    cv2.imwrite(output_folder + "usm.jpg", gray)
    # _, gray = preprocess(rect_img)
    # cv2.imwrite(output_folder + "preprocessed.jpg", gray)

    # 自适应阈值化，maxValue为255，基于邻域均值的二值化，反转，邻域大小，减去C调整
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      adp_threshold[0], adp_threshold[1])

    threshold *= parts_mask[:, :, 0]
    cv2.imwrite(output_folder + "thresh.jpg", threshold)

    # 在二值化图像中使原图绿色通道里<10的地方强制为0，去除白边
    threshold[rect_img[:, :, 1] < 10] = 0
    cv2.imwrite(output_folder + "binary.jpg", threshold)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori1, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "binary_result.jpg", ori1)



    # 获取椭圆元素，并开运算处理背景噪声 why 5?
    # kernel不同：1. 形状：方形利于保留边缘，圆形/椭圆利于消除噪声 2. 大小：小->更强的腐蚀和膨胀，消除噪声，影响形状 3. 方向
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(output_folder + "open.jpg", threshold)



    binary = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.uint8)
    rect_result = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.uint8)
    area_array = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.int16)


    # cv2.RETR_EXTERNAL 最外面的轮廓, cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，不进行压缩
    # contours是一个tuple， 里面是 n 个轮廓nparray，（点数 * 1 * 2） 表示
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori2, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "open_result.jpg", ori2)
    # cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)

    # 只有满足两个if的contour才会被认定为毛孔
    # count = 0
    times = 0
    contour_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20 and area < max_area:
            # ((x, y), (a, b), angle)
            ellipse = cv2.fitEllipse(cnt)
            short = int(ellipse[1][0])
            long = int(ellipse[1][1])
            if short / long > 0.2:
                
                contour_list.append(cnt)
                
                color = (0, 255, 0)
                thickness = 1
                # cv2.drawContours(mask, cnt, -1, color, thickness)
                
                points, bg_points = find_rect(cnt)
                # if is_region_of_interest(left_chin_points, points) or is_region_of_interest(right_chin_points, points):
                #     cv2.rectangle(mask, upper_left[::-1], lower_right[::-1], (0, 0, 255), 1)
                # if is_region_of_interest(left_chin_points, bg_points) or is_region_of_interest(right_chin_points, bg_points):
                #     cv2.rectangle(ori4, bg_points[0][::-1], bg_points[1][::-1], (0, 0, 255), 1)
                #     bg_points_list.append(bg_points)
                
                # flag = is_stain_BGR(stain_mask, bg_points)
                # if flag == True:
                #     color = (255, 0, 0)
                #     thickness = 2
                    # times += 1
                # if area > 200 and is_stain_HSV(img_hsv.copy()[bg_points[0][0]: bg_points[1][0]+1, bg_points[0][1]:bg_points[1][1]+1, ], bg_points):
                #     color = (0, 0, 255)
                #     thickness = 2
                #     # times += 1
                #     # cv2.putText(mask, str(times), (points[0][1], points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #     if is_stain_UV(rect_img_uv_hsv.copy()[bg_points[0][0]: bg_points[1][0]+1, bg_points[0][1]:bg_points[1][1]+1, ], bg_points):
                #         color = (255, 0, 0)
                #         thickness = 2
                #         times += 1
                # else:
                #     color = (0, 255, 0)
                #     thickness = 1

                # 对应的图片部分
                # cnt_rect = mask[points[0][0]:points[1][0], points[0][1]:points[1][1], :]
                # cnt_rect_bg = mask[bg_points[0][0]:bg_points[1][0], bg_points[0][1]:bg_points[1][1], :]

                # cv2.namedWindow('cnt_rect', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('cnt_rect', (200, 300))
                # cv2.imshow("cnt_rect", cnt_rect)

                # cv2.namedWindow('cnt_rect_bg', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('cnt_rect_bg', (400, 600))
                # cv2.imshow("cnt_rect_bg", cnt_rect_bg)
                # cv2.waitKey(0)

                # if area > 800:
                #     color = (0, 0, 255)
                #     thickness = 1
                #     area_area_flag = True
                #     times += 1
                # else:
                #     color = (0, 255, 0)
                #     thickness = 1
                #     area_area_flag = False

                # count += 1
                center_x = int(ellipse[0][0])
                center_y = int(ellipse[0][1])

                # why 0.6, 2, 15?
                radius = np.clip(int(0.6 * ellipse[1][0]), 2, 15)

                # 利用bianry记录拟合毛孔的椭圆半径, 但是没用上？+100的意义是什么
                # 一个物体的长短轴与矩阵是相反的
                # binary[center_y, center_x] = radius + 100
                # area_array[center_y, center_x] = area

                # dst， contour， all, white, thickness
                # color = (0, 255, 0)
                # thickness = 1
                cv2.drawContours(rect_img, cnt, -1, color, thickness)
                cv2.drawContours(rect_result, cnt, -1, 255, 1)
                # if area_area_flag == True:
                #     cv2.putText(mask, str(times), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


    print(f"detect {times} stains")

    rect_real_result = rect_img
    # print(radius_list.mean())
    cv2.imwrite(output_folder + "result.jpg", rect_real_result)

    return rect_result, rect_real_result, contour_list

def detect_head(rect_img, max_area, parts_mask):
    """同上detect，删去了uv作为参数"""
    output_folder = r"./temp/"
    rect_img = rect_img.copy()

    ori1, ori2, ori3, ori4 = rect_img.copy(), rect_img.copy(), rect_img.copy(), rect_img.copy()
    cv2.imwrite(output_folder + "ori.jpg", ori1)

    # _, gray = preprocess(rect_img)
    gray = pg_preprocess(rect_img)
    thresh_value = find_peak(gray)

    cv2.imwrite(output_folder + "preprocessed.jpg", gray)
    # 自适应阈值化，maxValue为255，基于邻域均值的二值化，反转，邻域大小，减去C调整
    # threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,5)
    ret, threshold = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    # ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    
    threshold *= parts_mask[:, :, 0]
    cv2.imwrite(output_folder + "thresh.jpg", threshold)

    # 在二值化图像中使原图绿色通道里<10的地方强制为0，去除白边
    # threshold[rect_img[:, :, 1] < 10] = 0
    # cv2.imwrite(output_folder + "binary.jpg", threshold)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori1, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "binary_result.jpg", ori1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(output_folder + "open.jpg", threshold)

    binary = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.uint8)
    rect_result = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.uint8)
    area_array = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.int16)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori2, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "open_result.jpg", ori2)

    times = 0
    contour_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20 and area < max_area:
            # ((x, y), (a, b), angle)
            ellipse = cv2.fitEllipse(cnt)
            short = int(ellipse[1][0])
            long = int(ellipse[1][1])
            if short / long > 0.2:
                
                contour_list.append(cnt)
                
                color = (0, 255, 0)
                thickness = 1
                # cv2.drawContours(mask, cnt, -1, color, thickness)
                
                points, bg_points = find_rect(cnt)
                # if is_region_of_interest(left_chin_points, points) or is_region_of_interest(right_chin_points, points):
                #     cv2.rectangle(mask, upper_left[::-1], lower_right[::-1], (0, 0, 255), 1)
                # if is_region_of_interest(left_chin_points, bg_points) or is_region_of_interest(right_chin_points, bg_points):
                #     cv2.rectangle(ori4, bg_points[0][::-1], bg_points[1][::-1], (0, 0, 255), 1)
                #     bg_points_list.append(bg_points)
                
                # flag = is_stain_BGR(stain_mask, bg_points)
                # if flag == True:
                #     color = (255, 0, 0)
                #     thickness = 2
                    # times += 1
                # if area > 200 and is_stain_HSV(img_hsv.copy()[bg_points[0][0]: bg_points[1][0]+1, bg_points[0][1]:bg_points[1][1]+1, ], bg_points):
                #     color = (0, 0, 255)
                #     thickness = 2
                #     # times += 1
                #     # cv2.putText(mask, str(times), (points[0][1], points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #     if is_stain_UV(rect_img_uv_hsv.copy()[bg_points[0][0]: bg_points[1][0]+1, bg_points[0][1]:bg_points[1][1]+1, ], bg_points):
                #         color = (255, 0, 0)
                #         thickness = 2
                #         times += 1
                # else:
                #     color = (0, 255, 0)
                #     thickness = 1

                # 对应的图片部分
                # cnt_rect = mask[points[0][0]:points[1][0], points[0][1]:points[1][1], :]
                # cnt_rect_bg = mask[bg_points[0][0]:bg_points[1][0], bg_points[0][1]:bg_points[1][1], :]

                # cv2.namedWindow('cnt_rect', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('cnt_rect', (200, 300))
                # cv2.imshow("cnt_rect", cnt_rect)

                # cv2.namedWindow('cnt_rect_bg', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('cnt_rect_bg', (400, 600))
                # cv2.imshow("cnt_rect_bg", cnt_rect_bg)
                # cv2.waitKey(0)

                # if area > 800:
                #     color = (0, 0, 255)
                #     thickness = 1
                #     area_area_flag = True
                #     times += 1
                # else:
                #     color = (0, 255, 0)
                #     thickness = 1
                #     area_area_flag = False

                # count += 1
                center_x = int(ellipse[0][0])
                center_y = int(ellipse[0][1])

                # why 0.6, 2, 15?
                radius = np.clip(int(0.6 * ellipse[1][0]), 2, 15)

                # 利用bianry记录拟合毛孔的椭圆半径, 但是没用上？+100的意义是什么
                # 一个物体的长短轴与矩阵是相反的
                # binary[center_y, center_x] = radius + 100
                # area_array[center_y, center_x] = area

                # dst， contour， all, white, thickness
                # color = (0, 255, 0)
                # thickness = 1
                cv2.drawContours(rect_img, cnt, -1, color, thickness)
                cv2.drawContours(rect_result, cnt, -1, 255, 1)
                # if area_area_flag == True:
                #     cv2.putText(mask, str(times), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


    print(f"detect {times} stains")


    rect_real_result = rect_img
    cv2.imwrite(output_folder + "result.jpg", rect_real_result)
    # print(radius_list.mean())
    return rect_result, rect_real_result, contour_list

def find_peak(gray):
    """查找当取前10%的点时，阈值应为多少"""
    unique_values, counts = np.unique(gray, return_counts=True)
    peak_percent = np.sum(counts)*0.1
    cumulative_count = 0
    for value, count in zip(unique_values, counts):
        cumulative_count += count
        if cumulative_count >= peak_percent:
            peak_value = value
            break

    mean = np.mean(gray)
    std = np.std(gray)
    # print(mean, std)

    plt.plot(unique_values, counts, marker='o')

    
    print(peak_value, mean+1*std)
    index = np.where(unique_values == peak_value)
    plt.plot(peak_value, counts[index], 'ro')  # 'ro'表示红色圆圈标记
    # plt.scatter(peak_value, counts[index], color='red', label='Peak Value')
    # plt.show()
    return peak_value

def watershed(gray):
    """后期可以考虑用分水岭变换优化"""
    pass

import cut
import os
import time
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parts_list = ["left_pore_area", "right_pore_area"]
    max_area = 2000
    input_folder = r"F:\fh\center_white\0411\selected\input/"
    file_list = sorted(os.listdir(input_folder))


    output_folder = "./output/" 
    count = 0
    for file in file_list:
        name = file.split(".")[0]
        # file = name
        img_parallel = cv2.imread(input_folder + file)
        # img_cross = cv2.imread(folder + f"/{file}/center/cross.jpg")
        # img_uv = uv_correct(cv2.imread(folder + f"/{file}/center/uv.jpg"))
        # img_white = cv2.imread(folder + f"/{file}/center/white.jpg")


        rect_img, parts_mask, xyxy, right_points, left_points, keypoint = cut.face_crop(img_parallel, parts_list)
        if len(keypoint) == 0:
            continue

        parts_img = rect_img * parts_mask
        
        # rect_img_uv = img_uv[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        # parts_img_uv = rect_img_uv * parts_mask
        # rect_img_cross = img_cross[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        # parts_img_cross = rect_img_cross * parts_mask
        # rect_img_white = img_white[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        # parts_img_white = rect_img_white * parts_mask



        start = time.time()
        rect_result, rect_real_result, contour_list = detect_head(rect_img, max_area, parts_mask)


        
        # plt.plot(unique_values, counts, marker='o')


        # rect_result_uv, rect_real_result_uv, contour_list_uv = detect(rect_img_uv, max_area, parts_mask, rect_img_uv)
        # rect_result_cross, rect_real_result_cross, contour_list_cross = detect(rect_img_cross, max_area, parts_mask, rect_img_cross)
        # rect_result_white, rect_real_result_white, contour_list_white = detect(rect_img_white, max_area, parts_mask, rect_img_white)
        end = time.time()
        print(end - start)

        cv2.imwrite(output_folder + name + "_parallel.jpg", img_parallel)
        cv2.imwrite(output_folder + name + "_pg_result0.10.jpg", rect_real_result)
        cv2.imwrite(output_folder + name + ".jpg", rect_img)


        # cv2.imwrite(f"./temp/{name}.jpg", rect_img)
        # cv2.imwrite(f"./temp/{name}_preprocessed.jpg", preprocessed)
        # cv2.imwrite("rect_result.jpg", rect_result)
        # cv2.imwrite("rect_real_result.jpg", rect_real_result)
        # seg_mask = make_seg_mask(parts_img, contour_list)
        # seg_mask_uv = make_seg_mask(parts_img_uv, contour_list_uv)
        # seg_mask_cross = make_seg_mask(parts_img_cross, contour_list_cross)
        # seg_mask_white = make_seg_mask(parts_img_white, contour_list_white)
        # # right_chin = seg_mask[right_chin_point[0][0]-xyxy[1]:right_chin_point[1][0]-xyxy[1], right_chin_point[0][1]-xyxy[0]:right_chin_point[1][1]-xyxy[0], :]
        # # left_chin = seg_mask[left_chin_point[0][0]-xyxy[1]:left_chin_point[1][0]-xyxy[1], left_chin_point[0][1]-xyxy[0]:left_chin_point[1][1]-xyxy[0], :]
        # right_chin_mask_uv = seg_mask_uv[right_points[0][1]-xyxy[1]:right_points[1][1]-xyxy[1], right_points[0][0]-xyxy[0]:right_points[1][0]-xyxy[0], :]
        # left_chin_mask_uv = seg_mask_uv[left_points[0][1]-xyxy[1]:left_points[1][1]-xyxy[1], left_points[0][0]-xyxy[0]:left_points[1][0]-xyxy[0], :]
        # right_chin_ori_uv = img_uv[right_points[0][1]:right_points[1][1], right_points[0][0]:right_points[1][0], :]
        # left_chin_ori_uv = img_uv[left_points[0][1]:left_points[1][1], left_points[0][0]:left_points[1][0], :]

        # right_chin_mask_cross = seg_mask_cross[right_points[0][1]-xyxy[1]:right_points[1][1]-xyxy[1], right_points[0][0]-xyxy[0]:right_points[1][0]-xyxy[0], :]
        # left_chin_mask_cross = seg_mask_cross[left_points[0][1]-xyxy[1]:left_points[1][1]-xyxy[1], left_points[0][0]-xyxy[0]:left_points[1][0]-xyxy[0], :]
        # right_chin_ori_cross = img_cross[right_points[0][1]:right_points[1][1], right_points[0][0]:right_points[1][0], :]
        # left_chin_ori_cross = img_cross[left_points[0][1]:left_points[1][1], left_points[0][0]:left_points[1][0], :]

        # right_chin_mask_white = seg_mask_white[right_points[0][1]-xyxy[1]:right_points[1][1]-xyxy[1], right_points[0][0]-xyxy[0]:right_points[1][0]-xyxy[0], :]
        # left_chin_mask_white = seg_mask_white[left_points[0][1]-xyxy[1]:left_points[1][1]-xyxy[1], left_points[0][0]-xyxy[0]:left_points[1][0]-xyxy[0], :]
        # right_chin_ori_white = img_white[right_points[0][1]:right_points[1][1], right_points[0][0]:right_points[1][0], :]
        # left_chin_ori_white = img_white[left_points[0][1]:left_points[1][1], left_points[0][0]:left_points[1][0], :]

        # right_chin_mask = seg_mask[right_points[0][1]-xyxy[1]:right_points[1][1]-xyxy[1], right_points[0][0]-xyxy[0]:right_points[1][0]-xyxy[0], :]
        # left_chin_mask = seg_mask[left_points[0][1]-xyxy[1]:left_points[1][1]-xyxy[1], left_points[0][0]-xyxy[0]:left_points[1][0]-xyxy[0], :]
        # right_chin_ori = img_parallel[right_points[0][1]:right_points[1][1], right_points[0][0]:right_points[1][0], :]
        # left_chin_ori = img_parallel[left_points[0][1]:left_points[1][1], left_points[0][0]:left_points[1][0], :]

        # # parts_real_result = rect_real_result * parts_mask
        # # parts_result = rect_result * parts_mask[:, :, 0]
        
        # # parts_real_result_uv = rect_img_uv * parts_mask
        # # cv2.drawContours(parts_real_result_uv, contour_list, -1, (255, 255, 255), 1)

        # # cv2.namedWindow("img_uv", cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow("img_uv", 500, 500)
        # # cv2.imshow("img_uv", parts_img_uv)
        # # cv2.waitKey(0)
        # # cv2.rectangle(img, left_points[0], left_points[1], (255, 0, 0), 5)
        # # cv2.rectangle(img, right_points[0], right_points[1], (255, 0, 0), 5)

        # # cv2.imwrite(output_foloder + name + "_seg.jpg", seg_mask)
        # cv2.imwrite(output_foloder + name + "_left_parallel_mask.jpg", left_chin_mask)
        # cv2.imwrite(output_foloder + name + "_right_parallel_mask.jpg", right_chin_mask)
        # cv2.imwrite(output_foloder + name + "_left_parallel.jpg", left_chin_ori)
        # cv2.imwrite(output_foloder + name + "_right_parallel.jpg", right_chin_ori)
        
        # cv2.imwrite(output_foloder + name + "_left_uv_mask.jpg", left_chin_mask_uv)
        # cv2.imwrite(output_foloder + name + "_right_uv_mask.jpg", right_chin_mask_uv)
        # cv2.imwrite(output_foloder + name + "_left_uv.jpg", left_chin_ori_uv)
        # cv2.imwrite(output_foloder + name + "_right_uv.jpg", right_chin_ori_uv)
        
        # cv2.imwrite(output_foloder + name + "_left_cross_mask.jpg", left_chin_mask_cross)
        # cv2.imwrite(output_foloder + name + "_right_cross_mask.jpg", right_chin_mask_cross)
        # cv2.imwrite(output_foloder + name + "_left_cross.jpg", left_chin_ori_cross)
        # cv2.imwrite(output_foloder + name + "_right_cross.jpg", right_chin_ori_cross)
        
        # cv2.imwrite(output_foloder + name + "_left_white_mask.jpg", left_chin_mask_white)
        # cv2.imwrite(output_foloder + name + "_right_white_mask.jpg", right_chin_mask_white)
        # cv2.imwrite(output_foloder + name + "_left_white.jpg", left_chin_ori_white)
        # cv2.imwrite(output_foloder + name + "_right_white.jpg", right_chin_ori_white)
        # # cv2.imwrite(output_foloder + name + "_result.jpg", parts_real_result)
        # # cv2.imwrite(output_foloder + name + ".jpg", img)
        # # cv2.imwrite(output_foloder + name + "_uv.jpg", parts_img_uv)
        # # cv2.imwrite(output_foloder + name + "_real_uv_result.jpg", parts_real_result_uv)