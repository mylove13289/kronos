import torch

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

# 输出应该是：tensor([1.], device='mps:0')

# 在训练时，将模型和数据移动到 MPS
#model = YourModel().to(mps_device)
#data = data.to(mps_device)

# 然后正常进行训练循环