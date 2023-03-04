import os

import ray
import torch
import ray_reducer
import time

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
n, m = 128, 4
backend = 'ray'
#backend = 'nccl'


def all_reduce_tensors(tensors):
  print(tensors)


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, rank, size):
        del os.environ['CUDA_VISIBLE_DEVICES'] 
        import ray_collectives
        import torch.distributed as dist
        self.rank = rank
        dist.init_process_group(backend, rank=rank, world_size=size)
        print(f"my rank is {rank}")
        torch.cuda.set_device(rank)
        
    def run(self):
        import torch.distributed as dist
        # warm up
        tensor = torch.rand(10 * 2 ** 20, dtype=torch.float64).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        start = time.time()
        for batch in range(10):
          print(f"batch {batch}")
          tensor = torch.rand(10 * 2 ** 20, dtype=torch.float64).cuda()
          dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
          print(tensor)
        elapsed = time.time() - start
        print(f"total time: {elapsed}")

ray.init(address="local")
ray_reducer.set_up_ray_reduce([(ray.get_runtime_context().get_node_id(), 4)])

actors = [Actor.remote(rank=i, size=4) for i in range(4)]
ray.get([actor.run.remote() for actor in actors])
