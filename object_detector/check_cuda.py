import torch
import torchvision

print(torch.__version__)  # Should show PyTorch 2.x
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)  # Should show CUDA 12.1
print(torchvision.ops.nms.__module__)  # Should show CUDA backend