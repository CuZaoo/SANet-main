import os
import random

import torchvision.transforms as transforms
import cv2 as cv2
import numpy as np
import torch

ignore_label = 19
label_mapping = {-1: ignore_label, 0: ignore_label,
                 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label,
                 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label,
                 10: ignore_label, 11: 2, 12: 3,
                 13: 4, 14: ignore_label, 15: ignore_label,
                 16: ignore_label, 17: 5, 18: ignore_label,
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15,
                 29: ignore_label, 30: ignore_label,
                 31: 16, 32: 17, 33: 18}


def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


color_map = [(128, 64, 128),
             (244, 35, 232),
             (70, 70, 70),
             (102, 102, 156),
             (190, 153, 153),
             (153, 153, 153),
             (250, 170, 30),
             (220, 220, 0),
             (107, 142, 35),
             (152, 251, 152),
             (70, 130, 180),
             (220, 20, 60),
             (255, 0, 0),
             (0, 0, 142),
             (0, 0, 70),
             (0, 60, 100),
             (0, 80, 100),
             (0, 0, 230),
             (119, 11, 32),
             ]
toPIL = transforms.ToPILImage()

img = cv2.imread('../../data/cityscapes/gtFine/aachen_000000_000019_gtFine_labelIds.png',
                 cv2.IMREAD_GRAYSCALE)

# img = img[:,:,::-1].transpose(2,0,1)
y_k_size = x_k_size  = 6
img = convert_label(img)
nums = 20
edges_f = cv2.Canny(img + 1, 0.1, 0.2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges_f[edges_f == 255] = 20

edges_f2 = cv2.dilate(edges_f,kernel, iterations=1,borderType=cv2.BORDER_REFLECT)
# 注意uint类型矩阵加法运算，溢出没有报错提醒
edges = edges_f2 + edges_f
edges[edges > 20] = 0
img_edges = img + edges
img_edges[img_edges < 20] = 0
img_edges[img_edges == 39] = 0
img_edges[img_edges >= 20] -= 19
img_edges = cv2.dilate(img_edges, kernel, iterations=2,borderType=cv2.BORDER_REFLECT)
img_edges[img_edges == 0 ] = 20
img_edges -= 1
sv_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
for i, color in enumerate(color_map):
    for j in range(3):
        sv_img[:, :, j][img_edges == i] = color_map[i][j]
img_PIL = toPIL(sv_img)  # 张量tensor转换为图片
# img_PIL.show()
sv_path = "../../segmentation_image/multi_class_boundary_detection/"
if not os.path.exists(sv_path):
    os.makedirs(sv_path)
img_PIL.save(f'{sv_path}edge.png')

# img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
# x = img_tensor.unsqueeze(0)
# print(x.size())
#
#
# label = cv.imread('../data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png')
# edge = cv.Canny(label, 0.1, 0.2)
#
# y_k_size = 6
# x_k_size = 6
# kernel = np.ones((4, 4), np.uint8)
# # 裁剪未标记的标签，从label中可以看出外边6层全是unlabel
# edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
# edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')
# img_PIL = toPIL(edge)  # 张量tensor转换为图片
# img_PIL.show()
# edge_d = cv.dilate(edge, kernel, iterations=1)
# img_PIL = toPIL(edge_d)  # 张量tensor转换为图片
# img_PIL.show()
# transf = transforms.ToTensor()
# x = transf(img)
# toPIL = transforms.ToPILImage()
# img_PIL = toPIL(x)  # 张量tensor转换为图片
# img_PIL.show()
# x = x.unsqueeze(0)
# x_size = x.size()
# im_arr = x.cpu().detach().numpy().transpose((0, 2, 3, 1))
# canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
# for i in range(x_size[0]):
#     canny[i] = cv.Canny(im_arr[i], 100, 200)
# canny = torch.from_numpy(canny).cuda().float()
# print(canny.size())
# toPIL = transforms.ToPILImage()
# canny = canny.squeeze(0)
# img_PIL = toPIL(canny)  # 张量tensor转换为图片
# img_PIL.show()  # 保存图片；img_PIL.show()可以直接显示图片
