import torch, torchvision
from torchvision.ops import nms

print(torch.__version__)
print(torch.version.cuda)
print(torchvision.__version__)

boxes = torch.tensor([[0,0,10,10],[1,1,11,11]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
print(nms(boxes, scores, 0.5))  # should run without error

"""
print(torch.__version__)         # should show +cu130
print(torch.version.cuda)        # should show 13.0
print(torch.cuda.is_available()) # should be True
print(torch.cuda.device_count()) # should be >0
print(torch.cuda.get_device_name(0)) # should print your GPU model

"""
