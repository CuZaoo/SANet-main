import os
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# 应用类激活
from pytorch_grad_cam import GradCAM
import time

# 更换运行时路径：设置父类为运行时路径
os.chdir("..")


# 更换任务名
save_father_path = "../segmentation_image"
# task_name = "motorcycle_pid_test"
task_name = "heat_map"
path = os.path.join(save_father_path, task_name)
# 更换模型
from models.sanet_S import get_pred_model
model_state_file = "../pretrained_models/cityscapes/sanet.pt"
# from models.ddrnet_23_slim2 import get_pred_model
# model_state_file = "best/ddrnet_23_slim.pth"



# 更换图片
# defaultPath = "data/cityscapes/leftImg8bit/test/"
defaultPath = "../data/cityscapes/leftImg8bit/"
imgPaths = os.listdir(defaultPath)
# imgPaths = ["bus1", "bus2", "bus3", "bus4","bus5","bus6"]
for (i, img) in enumerate(imgPaths):
    imgPaths[i] = defaultPath + img
# 更换类别
# 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#         'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
#         'train', 'motorcycle', 'bicycle'
# class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
#                'train', 'motorcycle', 'bicycle']
class_names = ['road']

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    # 载入权重
    model = get_pred_model()
    model = model.eval()
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    print(f'载入的权重数:{len(pretrained_dict)}')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()

    # model = SegmentationModelOutputWrapper(model)

    for (index, imgPath) in enumerate(imgPaths):
        """ 返送最后一层热力图 """
        file_path, file_name = os.path.split(imgPath)
        short_name, extension = os.path.splitext(file_name)
        image = np.array(Image.open(imgPath).convert('RGB'))
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # Taken from the torchvision tutorial
        # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html

        # 创建任务目录
        if not os.path.exists(path):
            os.makedirs(path)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        output = model(input_tensor)
        # print(type(output), output.keys())

        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
        sem_classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle']
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        for (idx, class_name) in enumerate(class_names):
            car_category = sem_class_to_idx[class_name]  # 更换类别
            car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
            car_mask_float = np.float32(car_mask == car_category)

            # 查看预测类别结果
            both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
            both_images = Image.fromarray(both_images)
            time_str = time.strftime('%d%H%M')
            img_path = os.path.join(path, short_name + "-" + class_name + "-预测类别结果-" + time_str + '.png')
            both_images.save(img_path)

            target_layers = [model.final_layer]  # 目标层
            targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
            with GradCAM(model=model,
                         target_layers=target_layers,
                         use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets)[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            res = Image.fromarray(cam_image)
            img_path = os.path.join(path, short_name + "-" + class_name + "-热力图-" + time_str + '.png')
            res.save(img_path)


if __name__ == '__main__':
    main()
