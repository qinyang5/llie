import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from collections import OrderedDict
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from  vismodel import low_light_transformer
import torch.nn as nn
# def getnfandnfg(img_LQ):
#     # img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
#     img_nf = img_LQ
#     img_nf = cv2.blur(img_nf, (5, 5))
#     img_nf_grad = cv2.Laplacian(img_nf.astype('uint8'), cv2.CV_8U, ksize=3)
#     img_nf_grad_lap = cv2.convertScaleAbs(img_nf_grad)
#     img_nf_grad_lap = img_nf_grad_lap / 255.0
#     img_nf_grad_lap = torch.Tensor(img_nf_grad_lap).float().permute(2, 0, 1)
#     # img_nf = cv2.blur(img_nf, (5, 5))
#     img_nf = img_nf * 1.0 / 255.0
#     img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)
#     return img_nf, img_nf_grad_lap
def load_network(load_path, network, strict=True):

    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    del load_net
    del load_net_clean
    return network
mymodel = low_light_transformer(64, predeblur=True, HR_in=True, w_TSA=True, front_RBs= 1,back_RBs=1).to(torch.device('cuda'))
mymodel = load_network('../../pretrained/prediction.pth', mymodel)
# resnet18 = models.resnet18(pretrained=True)

mymodel.eval()
target_layers = [mymodel.upconv1]
origin_img = cv2.imread('/models/archs/111.png')
rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128),
    transforms.CenterCrop(128)
])

crop_img = trans(rgb_img)
# 使用ImageNet数据集的均值和标准差。.unsqueeze(0)增加一个维度，使得张量变为batch形式，符合PyTorch模型输入要求。
# net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)
net_input = crop_img.unsqueeze(0)
# 乘以255是为了将这些值转换回[0, 255]区间，将张量的数据类型转换为8位无符号整型
'''
原本的张量或数组的形状是(channels, height, width)，这是常见的深度学习模型中的图像表示方式（通道优先）。
而OpenCV以及许多图像处理库通常期待的格式是(height, width, channels)（高度、宽度、通道顺序）。
因此，通过调用.transpose(1, 2, 0)，我们把通道维度从第0位移动到最后，使其适应OpenCV的显示需求。
'''
canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

cam = pytorch_grad_cam.GradCAMPlusPlus(model=mymodel, target_layers=target_layers)
grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]  # 从热力图数组中取出第一张图像的热力图

src_img = np.float32(canvas_img) / 255  # 将显示用的图像转换为浮点数并归一化到[0,1]区间，便于后续的热力图叠加。

# 在原始图像上叠加热力图。use_rgb=False表明热力图是以灰度形式叠加的，而非彩色。
visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
cv2.imshow('feature map', visualization_img)
cv2.waitKey(0)  # 等待用户按键，按任意键后关闭显示窗口