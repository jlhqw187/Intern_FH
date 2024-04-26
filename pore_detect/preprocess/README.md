## 文件介绍
1. cut.py及face_detector.py用于检测人脸关键点并分割出所需部位矩形图
2. pore_detect.py为测试时具体使用的代码，需要import cut和face_detector才可以用，写的很乱
3. preprocess.py是单独把图像预处理部分整理出来的代码。

## pore_detector.py三种方法的具体使用方式
### 原始的方法
1. 确保主函数使用的是detect()并且传入了四个参数（第四个参数用rect_img占位即可，是为了方便色斑检测的部分没有删去）
2. 其他都不需要动

### pg的方法
1. 确保主函数使用的是detect_head()并且传入了三个参数
2. 使用gray = pg_preprocess(rect_img)，注释掉_, gray = preprocess(rect_img)
3. 使用thresh_value = 120，ret, threshold = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
   注释掉threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,121,5)

### lep的方法
1. 确保主函数使用的是detect_head()并且传入了三个参数
2. 其他三点与pg相反

### 修改图片读取的方式
具体情况具体分析