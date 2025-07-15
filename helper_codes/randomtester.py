# def sample_indices(n, total_frames):
#     return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

# listy = sample_indices(8, 32)

# for idx, i in enumerate(listy):
#     listy[idx] -= 1
#     print(listy[idx])
import torch
import PIL.Image as Image
import numpy as np
import os
import sys
import copy


print("current working directory", os.getcwd())
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")
sys.path.append("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")

print("sys path", sys.path)

from custom_models import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
stem = copy.deepcopy(slow_model.blocks[0])
layer1 = copy.deepcopy(slow_model.blocks[1])
layer2 = copy.deepcopy(slow_model.blocks[2])
layer3 = copy.deepcopy(slow_model.blocks[3])
layer4 = copy.deepcopy(slow_model.blocks[4])
layer5 = copy.deepcopy(slow_model.blocks[5])
layer5.proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)

model = WeightedFocusNet(stem, layer1, layer2, layer3, layer4, layer5).to(device)

#Binary Mask: [B, C=1, T=8, H=32, W=32]

binary_mask = torch.ones((2, 1, 8, 32, 32)).to(device)  # Example binary mask
input = torch.zeros((2, 3, 8, 256, 256)).to(device)
out = model(input, binary_mask)




# input = torch.zeros((2, 512, 8, 32, 32)).to(device)
# model = HumanBackgroundWeighting(512).to(device)
# output = model(input)
# print(output)
# alpha = output[0][0].item()

# mask = torch.ones((1, 1, 8, 32, 32))
# mult = mask * alpha
# print("weighted mask shape", mult.shape)

# layer = torch.ones((1, 512, 8, 32, 32))
# final = layer * mult
# print("final shape", final.shape)




