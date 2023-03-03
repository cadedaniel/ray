import ray
import logging 
import torch

reduce_sequence = 0

def all_reduce_tensors(tensors):
  global reduce_sequence
  logging.info(f"received tensors {tensors}")
  for t in tensors:
    assert t.is_cuda
    all_reduce_impl1(t, reduce_sequence)
  reduce_sequence += 1


def all_reduce_impl(tensor, sequence):
  reducer_name = f"cli:{ray.get_runtime_context().get_node_id()}:{tensor.get_device()}" 
  logging.info(f"sending {tensor}, sequence: {sequence} to reducer: {reducer_name}")
  actor = ray.get_actor(reducer_name)
  return actor.allreduce.remote(tensor, sequence)


def all_reduce_impl1(gpu_buffer, sequence):        
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  cpu_tensor = gpu_buffer.to('cpu')
  reduced = ray.get(reducer.reduce.remote(
      cpu_tensor,
      client_name,
      sequence,
  ))
  # TODO nonblocking? otherwise this consumes the event loop
  gpu_buffer.copy_(reduced)
