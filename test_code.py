import os
import sys
import cv2
import csv
from tqdm import tqdm
import numpy as np
from keras.models import load_model


# 规范化图片大小和像素值
def get_inputs(filepath, src=[]):
    pre_x = []
    for s in src:
        s = os.path.join(filepath, s)
        input = cv2.imread(s)
        input = cv2.resize(input, (128, 128), interpolation=cv2.INTER_LANCZOS4)  # 处理图片尺寸,使用lanczos方法填充像素
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x) / 255.0
    return pre_x


def test(filepath):
    # 需要对测试结果输出，输出结果格式见result.csv

    # 加载模型h5文件
    model = load_model("./dropout_Adam_lanczos.h5")
    # 输出模型结构
    model.summary()

    # 新建一个列表保存预测图片的地址
    images = []
    # 获取每张图片的地址，并保存在列表images中
    for fn in tqdm(os.listdir(filepath)):
        if fn.endswith('jpg'):
            images.append(fn)
    # 调用函数，规范化图片
    pre_x = get_inputs(filepath, images)
    # 预测
    pre_y = model.predict_classes(pre_x)
    csv_file = open(r'result.csv', 'w', newline='')
    csv_write = csv.writer(csv_file, dialect='excel')
    csv_write.writerow(('file_name', 'class'))
    for i in tqdm(zip(images, pre_y)):
        csv_write.writerow((i[0], abs(i[1][0]-1)))
    print('测试完成')


# print(sys.argv)
if __name__ == '__main__':
    tmp = sys.argv
    if len(tmp) < 2:
        print('输入测试路径名')
    else:
        filepath = tmp[1]  # filepath为测试路径名（该路径名下有很多图片）
        print(filepath)
        test(filepath)
