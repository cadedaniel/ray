import os

import torch
import ray_collectives
import torch.distributed as dist

import ray_collectives

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

dist.init_process_group("ray", rank=0, world_size=1)

x = torch.ones(6).to("cuda")
dist.all_reduce(x)
y = x.cuda()
dist.all_reduce(y)

print(f"cpu allreduce: {x}")
print(f"cuda allreduce: {y}")

try:
    dist.broadcast(x, 0)
except RuntimeError:
    print("got RuntimeError when calling broadcast")
