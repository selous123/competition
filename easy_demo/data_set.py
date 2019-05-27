# encoding: utf-8

import os
import random
from rpi_define import *
import pandas as pd
from skimage import io
import csv
import numpy as np
import cv2


class DataSet(object):
    def __init__(self, images_dir, csv_file_path, shuffle=True):
        """
        初始化数据集
        :param images_dir: 图片目录地址
        :param csv_file_path: 标注数据的.csv文件地址
        :param shuffle: 是否打乱顺序
        """
        self.shuffle = shuffle
        self.csv_data = pd.read_csv(csv_file_path)
        self.csv_data["ImagePath"] = [os.path.join(images_dir, file_name) for file_name in self.csv_data['ImageName']]
        self.pool = list()
        self.num_epoch = -1

    def get_size(self):
        """
        获取数据集大小
        :return: 数据集大小
        """
        return self.csv_data.shape[0]

    def reset_pool(self):
        """
        初始化采样池
        :return: None
        """
        self.pool = [_ for _ in range(self.get_size())]
        if self.shuffle:
            random.shuffle(self.pool)

    @staticmethod
    def read_image(path):
        """
        读取图片数据
        :param path: 图片路径
        :return: 图像数据矩阵
        """
        return io.imread(path) / 255.0

    @staticmethod
    def id_to_onehot(category_id):
        """
        将分类ID转换为独热向量
        :param category_id: 分类ID
        :return: 独热向量
        """
        onehot = np.zeros([CATEGORY_COUNT])
        onehot[category_id - 1] = 1
        return onehot

    @staticmethod
    def onehot_to_id(onehot):
        """
        将独热向量转换为分类ID
        :param onehot: 独热向量
        :return: 分类ID
        """
        return np.argmax(onehot) + 1

    def _get_batch_indices(self, size):
        ndx_list = list()
        if size >= len(self.pool):
            remain = size - len(self.pool)
            ndx_list.extend(self.pool)
            self.reset_pool()
            self.num_epoch += 1
            ndx_list.extend(self._get_batch_indices(remain))
        elif size > 0:
            ndx_list.extend(self.pool[:size])
            self.pool = self.pool[size:]
        return ndx_list

    def get_batch(self, size):
        """
        获取一个批次的采样数据
        :param size: 批次大小
        :return: 图像数据和对应的标签
        """
        ndx_list = self._get_batch_indices(size)
        data = [DataSet.read_image(self.csv_data["ImagePath"][ndx]) for ndx in ndx_list]
        label = [DataSet.id_to_onehot(self.csv_data["CategoryId"][ndx]) for ndx in ndx_list]
        return data, label

    def __iter__(self):
        """
        遍历所有采样数据和对应的标签
        :return: 单个采样数据和对应的标签
        """
        for p, c in zip(self.csv_data["ImagePath"], self.csv_data["CategoryId"]):
            yield DataSet.read_image(p), DataSet.id_to_onehot(c), p, c


def split_train_and_val_data_set(src_csv_path, train_csv_path, val_csv_path):
    """
    从原始数据集中分离出训练集和验证集
    :param src_csv_path: 原始数据集标注.csv文件地址
    :param train_csv_path: 生成的训练集数据标注.csv文件地址
    :param val_csv_path: 生成的验证集数据标注.csv文件地址
    :return: None
    """
    file_train = open(train_csv_path, 'w', newline='')
    train_writer = csv.writer(file_train)
    train_writer.writerow(["ImageName", "CategoryId"])

    file_val = open(val_csv_path, 'w', newline='')
    val_writer = csv.writer(file_val)
    val_writer.writerow(["ImageName", "CategoryId"])

    csv_data = pd.read_csv(src_csv_path)

    categories = set()
    for i in range(csv_data.shape[0]):
        category = csv_data["CategoryId"][i]
        if (category not in categories) or (random.randint(0, 9) > 0):
            train_writer.writerow([csv_data["ImageName"][i], category])
            categories.add(category)
        else:
            val_writer.writerow([csv_data["ImageName"][i], category])


def convert_images_to_same_size(csv_path, src_image_dir, dst_image_dir):
    """
    将所有图片转换成相同的大小并存储到指定文件夹下
    :param csv_path: 标注数据的.csv文件地址
    :param src_image_dir: 原始图片所在的文件夹
    :param dst_image_dir: 存储到指定文件夹
    :return: None
    """
    csv_data = pd.read_csv(csv_path)
    for image_name in csv_data["ImageName"]:
        src_image_path = os.path.join(src_image_dir, image_name)
        dst_image_path = os.path.join(dst_image_dir, image_name)
        img = cv2.imread(src_image_path)
        output = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite(dst_image_path, output)


if __name__ == '__main__':
    # 原始数据集
    SRC_IMAGE_DIR = "/home/lrh/dataset/competition_nh/data"
    SRC_CSV_PATH = "/home/lrh/dataset/competition_nh/data.csv"

    # 处理成统一大小图片的数据集
    NORM_IMAGE_DIR = "/home/lrh/dataset/competition_nh/norm_data"
    TRAIN_CSV_PATH = "/home/lrh/dataset/competition_nh/train.csv"
    VAL_CSV_PATH = "/home/lrh/dataset/competition_nh/val.csv"

    if not os.path.exists(NORM_IMAGE_DIR):
        os.makedirs(NORM_IMAGE_DIR)
        convert_images_to_same_size(SRC_CSV_PATH, SRC_IMAGE_DIR, NORM_IMAGE_DIR)

    # 从原始数据集中生成训练集和验证集
    split_train_and_val_data_set(SRC_CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH)
