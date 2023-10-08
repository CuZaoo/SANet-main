# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import scipy.io as scio
from .base_dataset import BaseDataset
import cv2

class GtaV(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=11,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=960,
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(GtaV, self).__init__(ignore_label, base_size,
                                     crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()

        self.ignore_label = ignore_label

        self.color_list = self.set_color_list(os.path.join(self.root, 'gtav', 'mapping.mat'))

        self.class_weights = None

        self.bd_dilate_size = bd_dilate_size


    def set_color_list(self,path):
        matdata = scio.loadmat(path)
        mapping = matdata['cityscapesMap'] * 255
        return mapping.astype(int)
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

        return files

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root, 'gtav', item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        label = cv2.imread(os.path.join(self.root, 'gtav', item["label"]),
                           cv2.IMREAD_GRAYSCALE)

        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_pad=False,
                                             edge_size=self.bd_dilate_size, city=False,gtav=True)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))


if __name__ == '__main__':
    gtav = GtaV('../data/', 'list/gtav/train.lst')
    d = next(iter(gtav))
    print(d)
