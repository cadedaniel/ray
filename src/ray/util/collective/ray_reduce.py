import ray
import logging 
import torch
import asyncio

reduce_sequence = 0
async_reduce_sequence = 0

def all_reduce_tensors(tensors):
  global reduce_sequence
  logging.info(f"received tensors {tensors}")
  for t in tensors:
    assert t.is_cuda
    all_reduce_impl(t, reduce_sequence)
  reduce_sequence += 1


def all_reduce_tensors_async(tensors, callback):
  asyncio.run(all_reduce_tensors_async_helper(tensors, callback))

async def all_reduce_tensors_async_helper(tensors, callback):
  global async_reduce_sequence
  sequence = async_reduce_sequence
  async_reduce_sequence += 1
  print(f"received tensors {tensors}")
  for t in tensors:
    assert t.is_cuda
    await all_reduce_impl_async(t, sequence)
  callback()


def all_reduce_impl(gpu_buffer, sequence):        
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  cpu_tensor = gpu_buffer.to('cpu')
  reduced = ray.get(reducer.reduce.remote(
      cpu_tensor,
      client_name,
      sequence,
  ))

  gpu_buffer.copy_(reduced)


async def all_reduce_impl_async(gpu_buffer, sequence):        
  print(f"getting actors {sequence}")
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  print(f"copy to cpu")
  cpu_tensor = gpu_buffer.to('cpu')
  print(f"call reduce")
  reduced = await reducer.reduce.remote(
      cpu_tensor,
      client_name,
      sequence,
  )

  print(f"copy to gpu")
  gpu_buffer.copy_(reduced)
  print(f"done")
