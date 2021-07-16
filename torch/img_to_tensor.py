import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

##########  np与Tensor  ######################################
def np_to_tensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

##########  PIL与Tensor  ######################################
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()

def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    # return image.to(device, torch.float)
    return image

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


img_path = '/home/jerry/Desktop/1.png'
image = Image.open(img_path).convert('RGB')  # PIL读图片

image_PIL.save('test_result.png', quality=95)  # PIL写图片

# PIL show图片
plt.figure("img")
plt.imshow(image_PIL)
plt.show()