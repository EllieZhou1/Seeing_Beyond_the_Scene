# def sample_indices(n, total_frames):
#     return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

# listy = sample_indices(8, 32)

# for idx, i in enumerate(listy):
#     listy[idx] -= 1
#     print(listy[idx])
import torch

slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
print(slow_model)
