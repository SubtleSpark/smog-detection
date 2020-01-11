from keras.utils import Sequence
import numpy as np
import os
import cv2
from cv2 import imread, resize, cvtColor
import pyimagesearch.config as config
import random
import xml.etree.cElementTree as et


class DataGenerator(Sequence):
    '''
    dataset: 文件的路径列表，不是文件名
    '''

    def __init__(self, dataset, batch_size, imgShape=(128, 128), shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.imgShape = imgShape
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.dataset)

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for dataPath in batch_data:
            x, y = self.getOne(dataPath)
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)

    def getOne(self, dataPath):
        # get img
        x = cvtColor(resize(imread(dataPath), self.imgShape), cv2.COLOR_BGR2RGB)

        # get label
        y = [1, 0]
        # 构建xml的路径
        _, basename = os.path.split(dataPath)
        mainname, _ = os.path.splitext(basename)
        labelPath = os.path.sep.join([config.LABEL_PATH, mainname + '.xml'])
        if os.path.exists(labelPath):
            tree = et.parse(labelPath)
            root = tree.getroot()

            for Object in root.findall('object'):
                y = [0, 1]
                name = Object.find('name').text
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

        return np.array(x) / 255., np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)


if __name__ == '__main__':
    dataset_smog = []
    dataset_non_smog = []
    for filename in os.listdir(config.SMOG_PATH):
        dataset_smog.append(os.path.sep.join([config.SMOG_PATH, filename]))

    for filename in os.listdir(config.NON_SMOG_PATH):
        dataset_non_smog.append(os.path.sep.join([config.NON_SMOG_PATH, filename]))

    dataset = dataset_smog + dataset_non_smog
    random.shuffle(dataset)
    print(dataset[:10])
    DG = DataGenerator(dataset=dataset, batch_size=2, shuffle=True)
    x, y = DG.__getitem__(1)
    print(x.shape, y.shape)
    print(x[1, 1, 1, 1])
    # xml 测试
    # tree = et.parse(r"F:\data_set\smog_data\label\11000000001310000829_2018-10-10 10-40-28.xml")
    # root = tree.getroot()
    #
    # filename = root.find('filename').text
    # print(filename)
    #
    # for Object in root.findall('object'):
    #     name = Object.find('name').text
    #     print(name)
    #     bndbox = Object.find('bndbox')
    #     xmin = bndbox.find('xmin').text
    #     ymin = bndbox.find('ymin').text
    #     xmax = bndbox.find('xmax').text
    #     ymax = bndbox.find('ymax').text
