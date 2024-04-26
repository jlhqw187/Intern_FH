"""
数据集划分
input: data_folder:最后输出的数据集位置, img_folder:毛孔矩形图片文件夹, label_folder:毛孔掩码图片文件夹
output: 按照train/test--imgs/labels  20% test 划分的数据集 在data_folder处
"""
import os
import random
import shutil

data_folder = r"F:\fh\center_white\0411\selected\pred/data"
img_folder = r"F:\fh\center_white\0411\selected\pred\imgs"
label_folder = r"F:\fh\center_white\0411\selected\pred\labels"
os.makedirs(data_folder, exist_ok=True)

img_files = os.listdir(img_folder)
random.shuffle(img_files)

size = len(img_files)
train_size = int(size * 0.8)
val_size = size - train_size

train_folder = os.path.join(data_folder, "train")
test_folder = os.path.join(data_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_imgs_folder = os.path.join(train_folder, "imgs")
train_labels_folder = os.path.join(train_folder, "labels")
test_imgs_folder = os.path.join(test_folder, "imgs")
test_labels_folder = os.path.join(test_folder, "labels")
os.makedirs(train_imgs_folder, exist_ok=True)
os.makedirs(test_imgs_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(test_labels_folder, exist_ok=True)

for i, img_file in enumerate(img_files):
    src_img = os.path.join(img_folder, img_file)
    src_label = os.path.join(label_folder, img_file.replace(".png", "_label.png"))  
    if i < train_size:
        dst_img = os.path.join(train_imgs_folder, img_file)
        dst_label = os.path.join(train_labels_folder, img_file.replace(".png", "_label.png"))
    else:
        dst_img = os.path.join(test_imgs_folder, img_file)
        dst_label = os.path.join(test_labels_folder, img_file.replace(".png", "_label.png"))
    shutil.copy(src_img, dst_img)
    shutil.copy(src_label, dst_label)