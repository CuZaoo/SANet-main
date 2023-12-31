#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


class Visualizer(object):
    def __init__(self, rect_size):
        self.rect_size = rect_size
        self.line_width = 1

    def fill_pixel(self, i, j, fill_color):
        self.img[j][i] = fill_color

    def draw_rect(self, i, j, fill_color):
        cv2.rectangle(self.img, (i * self.rect_size + self.line_width, j * self.rect_size + self.line_width), \
                      ((i + 1) * self.rect_size - self.line_width, (j + 1) * self.rect_size - self.line_width),
                      fill_color, -1)

    def color_index(self, num):
        if num < 256:
            return [num, 0, 0]
        elif num < 512:
            return [255, num - 256, 0]
        elif num < 768:
            return [768 - num, 256, 0]
        elif num < 1024:
            return [0, 256, num - 768]
        elif num < 1280:
            return [0, 0, 1280 - num]
        else:
            print("Color Error!")
            return [255, 255, 255]

    def visual_pixel(self, feat_map):
        input_ = feat_map.data
        self.img_size = input_.shape[0]
        size = self.img_size
        self.img = np.ones([size, size, 3]).astype(np.uint8) * 255
        colors = {}
        for i in range(self.img_size):
            for j in range(self.img_size):
                colors[input_[i][j]] = 1
        color_num = len(colors.keys())
        print('Different Color Num: ', color_num)
        color_step = int(1280 / (color_num + 1))
        colors = sorted(colors.items(), key=lambda item: item[0])
        color_dict = {0: 0}
        num = 1
        for color in colors:
            if color[0] != 0:
                color_dict[color[0]] = num * color_step
                num += 1
        print('Different Color Num: ', len(color_dict.keys()))
        for i in range(self.img_size):
            for j in range(self.img_size):
                color = self.color_index(color_dict[input_[i][j]])
                self.img.itemset((i, j, 0), color[0])
                self.img.itemset((i, j, 1), color[1])
                self.img.itemset((i, j, 2), color[2])
                # self.img[i][j][0] = color[0]
                # self.img[i][j][1] = color[1]
                # self.img[i][j][2] = color[2]

    def visual(self, feat_map):
        # if 'data' in feat_map:
        input_ = feat_map['data']
        # else:
        # input_ = feat_map.data
        print("rf: ", input_.shape)
        self.img_size = input_.shape[0]
        size = self.img_size * self.rect_size
        self.img = np.zeros([size, size, 3]).astype(np.uint8)
        cv2.rectangle(self.img, (0, 0), (size, size), (255, 255, 255), 2)
        colors = {}
        for i in range(self.img_size):
            for j in range(self.img_size):
                colors[input_[i][j]] = 1
        color_num = len(colors.keys())
        print('Different Color Num: ', color_num)
        # color_step = int(1920 * 128 / (color_num + 1))
        color_step = int(2345+1 / (color_num + 1))
        colors = sorted(colors.items(), key=lambda item: item[0])
        color_dict = {0: 0}
        num = 1
        for color in colors:
            if color[0] != 0:
                color_dict[color[0]] = num * color_step
                num += 1
        color_max = 0
        color_min = 1280
        for key in color_dict:
            if color_dict[key] > color_max:
                color_max = color_dict[key]
            if color_dict[key] < color_min:
                color_min = color_dict[key]

        if len(color_dict) == 2:
            for key in color_dict.keys():
                color_dict[key] = 640
        else:
            for key in color_dict.keys():
                temp = int(float(color_dict[key] - color_min) / float(color_max - color_min) * 1150)
                color_dict[key] = temp

        for i in range(self.img_size):
            for j in range(self.img_size):
                self.draw_rect(i, j, fill_color=self.color_index(color_dict[input_[i][j]]))

    def show(self):
        cv2.imshow('visual', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, path):
        cv2.imwrite(path, self.img)

    def size(self):
        return self.img_size
