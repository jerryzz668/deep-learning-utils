import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image


training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),#
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(2048),
                                         transforms.CenterCrop(2048),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)
# print(model)
print(type(model))

import torchsummary 
torchsummary.summary(model.cuda(),(3,224,224))

# save the conv layer weights in this list
model_weights =[]
# save the 49 conv layers in this list
conv_layers = []

# get all the model children as list
model_children = list(model.children())
counter = 0  # num of conv layers


# 根据resnet18，把相应的conv layers 和 wights 加入到 list中
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
# print(f"conv_layers: {conv_layers}")
model = model.to(device)
# 根据resnet18，把相应的conv layers 和 wights 加入到 list中


image = Image.open('/home/jerry/Desktop/garbage/1_test_demo/bbaug-demo/0191-0096-01_4281_2912.jpg')
#
image = testing_transforms(image)
print(f"Image shape before: {image.shape}")  # [3, 224, 224]
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")  # [1, 3, 224, 224]
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
#print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)

processed = []
for feature_map in outputs:
    # print('feature_map', feature_map.size())
    feature_map = feature_map.squeeze(0)  # [1, 64, 112, 112] --> [64, 112, 112]
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]  # 求64个channel的平均
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=10)
plt.show()
# plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

# for i in range(len(processed)):
#     cv2.imshow('img', processed[i])
#     cv2.waitKey(3000)

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(30, 50))
# for i in range(len(processed)):
#     plt.imshow(processed[i])
#     plt.show()