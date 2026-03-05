import torch
ckpt = torch.load("trained_models/kitti_8/error_resume.pt", map_location="cpu", weights_only=False)
ckpt["epoch"] = 49
torch.save(ckpt, "")