import ray
import logging 
import torch
import asyncio

reduce_sequence = 0
async_reduce_sequence = 0

def all_reduce_tensors(tensors):
  global reduce_sequence
  logging.info(f"all_reduce_tensors received tensors {tensors}")
  for t in tensors:
    assert t.is_cuda
    all_reduce_impl(t, reduce_sequence)
  reduce_sequence += 1


def all_reduce_tensors_async(tensors, callback):
  print(f'all_reduce_tensors_async')
  # TODO need to lift this into it's own thread
  asyncio.run(all_reduce_tensors_async_helper(tensors, callback))

async def all_reduce_tensors_async_helper(tensors, callback):
  global async_reduce_sequence
  print(f"received {len(tensors)} tensors")
  for t in tensors:
    assert t.is_cuda
    await all_reduce_impl_async(t, async_reduce_sequence)
    async_reduce_sequence += 1
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
  print(f'all_reduce_impl_async for {gpu_buffer.data_ptr()} seq {sequence}')
  print(f"getting actors {sequence}")
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  print(f"copy to cpu")
  cpu_tensor = gpu_buffer.to('cpu')
  print(f"call reduce")
  reduced = await reducer.reduce.remote(

  #print(f"call reduce (skipping completion)")
  #reduced = reducer.reduce.remote(
      cpu_tensor,
      client_name,
      sequence,
  )

  print(f"copy to gpu")
  gpu_buffer.copy_(reduced)
  print(f"done")
