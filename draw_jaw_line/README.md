## 任务
## input 原始人脸图片
## output 人脸下颌线
## sample里是我们方法的一个示例
## vplus0417里4张示例图片+1张修改后的图片
## 我们的方法

0. 要求人脸需佩戴耳罩，脖罩，否则把脖子涂黑如stained.jpg所示。
1. 获取人脸关键点，选取种子点skin_point和后面判断下颌线高度的jaw_line_points
2. 在种子点上做floodfill获取mask
3. 截取jaw_line_points最大y轴作为mask的起始点，起始点以上涂黑
4. findcontours找出jaw部分的contour
5. 用polylines画出不闭合的下颌线