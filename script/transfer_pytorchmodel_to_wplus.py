import torch
import os.path as osp

ckpt = "./experiments_wild/checkpoint-20000"

sd = torch.load(osp.join(ckpt,'pytorch_model.bin'), map_location="cpu")
w_proj_sd = {}
wplus_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("w_proj_model"):
        w_proj_sd[k.replace("w_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        wplus_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save({"w_proj": w_proj_sd, "wplus_adapter": wplus_sd}, osp.join(ckpt, "wplus_adapter.bin"))

num_params = 0
for param in w_proj_sd.values():
    num_params += param.numel()
a = num_params / 1e6
num_params = 0
for param in wplus_sd.values():
    num_params += param.numel()
b = num_params / 1e6
print(['proj: ',a , 'w-attn', b])
