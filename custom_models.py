import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

#Human Seg Branch ----->
#                       |--> Sum Concat (after 2nd layer)-->3 more layers --> output
#Original Branch ------>
class SumConcat(nn.Module):
    def __init__ (self):
        super().__init__()
        self.slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        self.orig_stem = copy.deepcopy(self.slow_model.blocks[0])
        self.orig_layer1 = copy.deepcopy(self.slow_model.blocks[1])
        self.orig_layer2 = copy.deepcopy(self.slow_model.blocks[2])

        self.seg_stem = copy.deepcopy(self.slow_model.blocks[0])
        self.seg_layer1 = copy.deepcopy(self.slow_model.blocks[1])
        self.seg_layer2 = copy.deepcopy(self.slow_model.blocks[2])

        self.layer3 = self.slow_model.blocks[3]
        self.layer4 = self.slow_model.blocks[4]
        self.layer5 = self.slow_model.blocks[5]
        self.layer5.proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)
    
    def forward (self, orig_img, seg_img):
        output_orig = self.orig_stem(orig_img)
        output_orig = self.orig_layer1(output_orig)
        output_orig = self.orig_layer2(output_orig)

        output_seg = self.seg_stem(seg_img)
        output_seg = self.seg_layer1(output_seg)
        output_seg = self.seg_layer2(output_seg)

        concat = output_orig + output_seg 

        out = self.layer3(concat)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
    


#Human Seg Branch ----->
#                       |--> Stack Concat (after 2nd layer)-->3 more layers --> output
#Original Branch ------>
class StackConcat(nn.Module):
    def __init__ (self):
        super().__init__()
        slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)

        self.orig_stem = copy.deepcopy(slow_model.blocks[0])
        self.orig_layer1 = copy.deepcopy(slow_model.blocks[1])
        self.orig_layer2 = copy.deepcopy(slow_model.blocks[2])

        self.seg_stem = copy.deepcopy(slow_model.blocks[0])
        self.seg_layer1 = copy.deepcopy(slow_model.blocks[1])
        self.seg_layer2 = copy.deepcopy(slow_model.blocks[2])

        self.layer3 = slow_model.blocks[3]

        #change input channels of these from 512 to 1024 bc we stacked
        self.layer3.res_blocks[0].branch1_conv = nn.Conv3d(1024, 1024, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False)
        self.layer3.res_blocks[0].branch2.conv_a = nn.Conv3d(1024, 256, kernel_size=(3, 1, 1), 
                                                                        stride=(1, 1, 1), padding=(1, 0, 0), bias=False)

        self.layer4 = slow_model.blocks[4]
        self.layer5 = slow_model.blocks[5]
        self.layer5.proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)
    
    def forward (self, orig_img, seg_img):
        output_orig = self.orig_stem(orig_img)
        output_orig = self.orig_layer1(output_orig)
        output_orig = self.orig_layer2(output_orig)

        output_seg = self.seg_stem(seg_img)
        output_seg = self.seg_layer1(output_seg)
        output_seg = self.seg_layer2(output_seg)

        concat = torch.cat([output_orig, output_seg], dim=1)
        out = self.layer3(concat)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


############################# VERSION 1 OF WEIGHTED FOCUS NET ###############################

#Given the output of ResStage 2, we want to predict delta_a and delta_b
# this will be used for the weighting for human and background
class HumanBackgroundWeighting1(nn.Module):
    #In channels: ([B, C=512, T=8, H=32, W=32])
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv= nn.Conv3d(in_channels, 16, kernel_size=1)
        self.linear1 = nn.Linear(16*8*4*4, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 2)

        #to make alpha and beta be 1 initially
        nn.init.constant_(self.linear2.bias, 0.0)
        nn.init.constant_(self.linear2.weight, 0.0)

    #x is the output of ResStage 2
    #x dimensions: [B, C=512, T=8, H=32, W=32]
    def forward(self, x):
        out = self.conv(x) # [B, C=16, T=8, H=32, W=32], reduce channels
        out = F.adaptive_avg_pool3d(out, output_size=(8, 4, 4))  # [B, 16, 8, 4, 4], make 2048 neurons
        out = out.view(out.size(0), -1)  # Flatten everything except batch dim
        out = self.linear1(out) # [B, hidden_channels=64]
        out = F.relu(out)  # [B, hidden_channels=64]
        out = self.linear2(out)
        return out



#This version clips values of alpha and beta
class WeightedFocusNet1(nn.Module):
    def __init__ (self):
        super().__init__()
        self.slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        self.slow_model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)
        self.ab_deltas = HumanBackgroundWeighting1(in_channels=512, hidden_channels=64)

    def forward (self, inputs, binary_mask):
        x = inputs
        for i, block in enumerate(self.slow_model.blocks):
            if i == 3:
                ab_delta = self.ab_deltas(x) #[B, 2]
                alpha_delta = ab_delta[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) #[B, 1, 1, 1, 1]
                beta_delta = ab_delta[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) # [B, 1, 1, 1, 1]
                
                alpha = 1.0 + alpha_delta
                beta = 1.0 + beta_delta
                x = alpha * binary_mask * x + beta * (1-binary_mask) * x

            x = block(x)
        return alpha, beta, x
    

############################# VERSION 2 OF WEIGHTED FOCUS NET ###############################

#Given the output of ResStage 2, we want to predict "ratio" which is a value that determines 
class HumanBackgroundWeighting2(nn.Module):
    #In channels: ([B, C=512, T=8, H=32, W=32])
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv= nn.Conv3d(in_channels, 16, kernel_size=1)
        self.linear1 = nn.Linear(16*8*4*4, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)

        #to make alpha and beta be 1 initially
        nn.init.constant_(self.linear2.bias, 0.0)
        nn.init.constant_(self.linear2.weight, 0.0)

    #x is the output of ResStage 2
    #x dimensions: [B, C=512, T=8, H=32, W=32]
    def forward(self, x):
        out = self.conv(x) # [B, C=16, T=8, H=32, W=32], reduce channels
        out = F.adaptive_avg_pool3d(out, output_size=(8, 4, 4))  # [B, 16, 8, 4, 4], make 2048 neurons
        out = out.view(out.size(0), -1)  # Flatten everything except batch dim
        out = self.linear1(out) # [B, hidden_channels=64]
        out = F.relu(out)  # [B, hidden_channels=64]
        out = self.linear2(out) #[B, 1]
        out = F.sigmoid(out) * 2.0 - 1.0  # Scale to [-1, 1]
        return out

#This version clips values of alpha and beta
class WeightedFocusNet2(nn.Module):
    def __init__ (self):
        super().__init__()
        self.slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        self.slow_model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)
        self.ratios = HumanBackgroundWeighting2(in_channels=512, hidden_channels=64)

    def forward (self, inputs, binary_mask):
        x = inputs
        for i, block in enumerate(self.slow_model.blocks):
            if i == 3:
                ratio = self.ratios(x) #[B, 2]
                ratio = ratio[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) #[B, 1, 1, 1, 1]

                alpha = 1.0 + ratio
                beta = 1.0 - ratio
                x = alpha * binary_mask * x + beta * (1-binary_mask) * x

            x = block(x)
        return alpha, beta, x