import torch
import torch.nn as nn

#Human Seg Branch ----->
#                       |--> Sum Concat (after 2nd layer)-->3 more layers --> output
#Original Branch ------>
class SumConcat(nn.Module):
    def __init__ (self, orig_stem, orig_layer1, orig_layer2, seg_stem, seg_layer1,
                  seg_layer2, layer3, layer4, layer5):
        super().__init__()
        self.orig_stem = orig_stem
        self.orig_layer1 = orig_layer1
        self.orig_layer2 = orig_layer2
        self.seg_stem = seg_stem
        self.seg_layer1 = seg_layer1
        self.seg_layer2 = seg_layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
    
    def forward (self, orig_img, seg_img):
        output_orig = self.orig_stem(orig_img)
        output_orig = self.orig_layer1(output_orig)
        output_orig = self.orig_layer2(output_orig)

        output_seg = self.segstem(seg_img)
        output_seg = self.seg_layer1(output_seg)
        output_seg = self.seg_layer2(output_seg)

        concat = output_orig + output_seg 

        out = self.layer3(concat)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
