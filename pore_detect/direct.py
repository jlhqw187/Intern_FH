"""
直接对输入进来的图片应用毛孔检测算法
input: 输入图片文件夹
output：输出文件夹
"""
import pore_detect
import numpy as np
import cv2
import os

def find_roi(img):
    gray = cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY)
    ret, roi_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    return roi_mask


def direct_detect(img):
    output_folder = r"./temp/"
    img = img.copy()
    ori1, ori2, ori3, ori4 = img.copy(), img.copy(), img.copy(), img.copy()

    gray = pore_detect.pg_preprocess(img)
    cv2.imwrite(output_folder + "preprocessed.jpg", gray)

    thresh_value = pore_detect.find_peak(gray)
    # threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,5)
    ret, threshold = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    # ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori1, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "binary_result.jpg", ori1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(output_folder + "open.jpg", threshold)

    result = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori2, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_folder + "open_result.jpg", ori2)

    times = 0
    contour_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 40 and area < 400:
            # ((x, y), (a, b), angle)
            ellipse = cv2.fitEllipse(cnt)
            short = int(ellipse[1][0])
            long = int(ellipse[1][1])
            if short / long > 0.2:
                
                contour_list.append(cnt)
                
                color = (0, 255, 0)
                thickness = 1
                # cv2.drawContours(mask, cnt, -1, color, thickness)
                
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
                cv2.drawContours(img, [cnt], -1, color, -1)
                # cv2.drawContours(result, cnt, -1, 255, -1)
                cv2.fillPoly(result, [cnt], (255, 255, 255))
                # if area_area_flag == True:
                #     cv2.putText(mask, str(times), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


    print(f"detect {times} stains")


    real_result = img
    cv2.imwrite(output_folder + "result.jpg", real_result)

    return result, real_result, contour_list

if __name__ == "main": 
    input_path = r"F:\fh\center_white\0412"
    output_path = r"F:\fh\center_white\0412\others_waited\others_waited_direct_output/"
    file_list = os.listdir(input_path)
    for file in file_list:
        name = file.split(".")[0]
        img = cv2.imread(input_path + file)
        # roi_mask = find_roi(img)
        # roi_image = cv2.bitwise_and(img, img, mask=roi_mask)
        result, real_result, contour_list = direct_detect(img)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)

        # cv2.imshow("result", real_result)
        # cv2.waitKey(0)
        cv2.imwrite(output_path + name + ".jpg", img)
        cv2.imwrite(output_path + name + "_result.jpg", result)
        cv2.imwrite(output_path + name + "_real_result.jpg", real_result)