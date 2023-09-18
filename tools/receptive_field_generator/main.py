#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ops import *
from visual import Visualizer


# from __future__ import print_function

def convs(num):
    print("conv")
    # Create Path
    path = os.getcwd()
    if not os.path.exists(os.path.join(path, 'conv')):
        os.mkdir(os.path.join(path, 'conv'))
    # Init Input
    h0 = {}
    h0['data'] = np.array([[1]])
    h0['stride'] = 1
    # SANet:stom-L6-APPPM
    # kernel = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 9, 17]
    # stride = [2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 4, 8]
    # SANet:stom-L6
    kernel = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    stride = [2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1]
    # SANet:stom-L3
    # kernel = [3, 3, 3, 3, 3, 3, 3, 3]
    # stride = [2, 2, 1, 1, 2, 1, 1, 1]

    # Stack Conv Layers and visual&save
    for i in range(len(stride)):
        tmp = conv(h0, kernel=kernel[i], stride=stride[i])
        h0 = tmp
        v = Visualizer(10)
        v.visual(tmp)
        v.save(os.path.join(path, 'conv', str(i) + '.jpg'))
        print('conv: ', v.size())
    return h0


def dilated_convs(h0):
    # Create Path
    path = os.getcwd()
    if not os.path.exists(os.path.join(path, 'dconv')):
        os.makedirs(os.path.join(path, 'dconv'))

    # Dilate Rate of Each Layer
    # SANet:DP1-DP2
    r = [1, 2, 5, 7, 13]

    # Init Input
    # h0 = {}
    # h0['data'] = np.array([[1]])
    # h0['stride'] = 1

    # Stack Dilated Conv Layers
    for i in range(len(r)):
        tmp = dilated_conv(h0, rate=r[i])
        h0 = tmp
        v = Visualizer(10)
        v.visual(tmp)
        img_path = os.path.join(path, 'dconv', str(i + 1) + '.jpg')
        print(img_path)
        v.save(img_path)
        print('dconv: ', v.size())


def example():
    net = []
    h0 = {}
    h0['data'] = np.array([[2]])
    h0['stride'] = 1

    h1 = conv(h0)
    # h2 = conv(h1)
    # h3 = dilated_conv(h2)
    # h4 = dilated_conv(h3)

    rect_size = 20
    v = Visualizer(rect_size)
    v.visual(h1)
    v.save(os.path.join('.', 'res.jpg'))




if __name__ == '__main__':
    print(os.getcwd())
    h0 = convs(10)
    # dilated_convs(h0)
    # example()
    # test()
