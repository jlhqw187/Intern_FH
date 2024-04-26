# 总体工作PIPELINE：
# 0. 筛选出光照均匀，毛孔较为明显的人脸图像
# 1. 运行pore_detect.py获取算法预测的毛孔掩膜result和real_result.(此时毛孔基本正确，但是会漏标一些并且有一些小错误（比如很小的色斑)）
# 2. 在real_result上用画图3D重新标注，主要是补充漏标并修改小错误
# 3. 运行extract_mask.py得到新的label（即result）
# 4. 运行dataset.py得到划分好的数据集
# 5. 使用语义分割模型进行训练，获取pred
# 6. 回到2，形成loop

## pore detect
## input: 一张白光或者平行光下的人脸照片
## output： 毛孔区域矩形的最终结果 
## 在结果中：result表示只含最后结果的掩码， real_result表示蒙在原始图片上的掩码，rect_img表示包含目标区域的最小矩形（类比min_rect）
## 注意：所有的图片需要无损保存，例cv2.imwrite(output_path + name + part_name + "_real_result.png", real_result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

## input文件夹里的first下是最初的实验图片，其他图片为具有代表性的人脸图片。
## preprocess文件夹中是具体会采用到的几种预处理的方法，其中有更详细的README。
## toolkits文件夹中是一些小的实用的工具。augumentation.py是使用了gamma校正和hsv增强的类，dataset.py用来划分数据集为8：2，extract_mask.py用来提取清理过数据的real_result上的掩码转换成result(即图像的label), crop_triangle.py是用来获取parts_jpg中心的1/3处的正方形并保存。

## baseline.py是最开始的算法 usm -> adaptive_thresh -> MORPH_OPEN -> filter
## cut.py 分割人脸parts的算法，相较于原始版本，1. 增加了切 left_pore_area, right_pore_area的方法 2. get_parts里多返回了一个参数xyxy，表示min_rect的左上角点和右下角点。
## direct_pipeline.py直接从原始人脸图片切出包含parts的min_rect的图片triangle_image, 只有掩码的resutl， 盖在原始图片上的real_result。
## direct.py 直接从triangle_image里进行检测，输出result, real_result
## face_detector.py是cut.py的module，用于特征点检测和人脸分割
## pore_detect.py 截止4.25时的检测算法

首先通过cut.face_crop()获取rect_img和parts_mask,两者相乘得只含parts的img（最小外接矩形）。
再把rect_img和parts_mask送到detect_head里去预测，pg_preprocess -> threshold(threh_value采用find_peak_value里的前10%的点) -> MORPH_OPEN -> filter(40 < area < 400)