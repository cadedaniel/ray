import ray
import logging 
import torch
import asyncio
import time
from collections import defaultdict
import threading
import os

reduce_sequence = 0
async_reduce_sequence = 0

class Ctx:
    def __init__(self, sequence, rank, stage_tracker):
        self.sequence = sequence
        self.stage_tracker = stage_tracker
        self.rank = rank
        self.zero_time = time.time()
        self.last_event_time = self.zero_time

    def swap_last_event_time(self, t):
        last_event_time = self.last_event_time
        self.last_event_time = t
        return last_event_time

def all_reduce_tensors_async(tensors, callback):
  print(f'all_reduce_tensors_async, called from thread {threading.get_native_id()} pid {os.getpid()}')
  # TODO need to lift this into it's own thread
  asyncio.run(all_reduce_tensors_async_helper(tensors, callback))

async def all_reduce_tensors_async_helper(tensors, callback):
  global async_reduce_sequence
  start_t = time.time()
  print(f"all_reduce_tensors_async_helper received {len(tensors)} tensors, thread {threading.get_native_id()}")
  for t in tensors:
    assert t.is_cuda
    ctx = stage_tracker.create_ctx(async_reduce_sequence)
    await all_reduce_impl_async(t, async_reduce_sequence, ctx)
    async_reduce_sequence += 1
  log_with_time_no_rank('all_reduce_tensors_async_helper', time.time() - start_t, -1)

  if stage_tracker.should_print(async_reduce_sequence) and ctx.rank == 0:
    stage_tracker.print('client')

  callback()


async def all_reduce_impl_async(gpu_buffer, sequence, ctx):        
  print(f'all_reduce_impl_async for {gpu_buffer.data_ptr()} seq {sequence}, size_bytes={gpu_buffer.element_size() * gpu_buffer.nelement()}')
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  log_with_time(f"got actors", ctx)

  cpu_tensor = gpu_buffer.to('cpu')
  log_with_time(f"copied to cpu", ctx)
  reduced = await reducer.reduce.remote(

  #print(f"call reduce (skipping completion)")
  #reduced = reducer.reduce.remote(
      cpu_tensor,
      client_name,
      sequence,
      ctx.rank,
  )
  log_with_time(f"called reduce", ctx)

  gpu_buffer.copy_(reduced)
  log_with_time(f"copied to gpu", ctx)

class StageTracker:
    def __init__(self):
        self.stages = defaultdict(list)

    def on_log(self, message, dur_since_last_event_s):
        self.stages[message].append(dur_since_last_event_s * 1000)

    def create_ctx(self, sequence, rank=None):
        if rank == None:
            import os
            rank = int(os.environ['RANK'])
        else:
            rank = rank

        return Ctx(sequence, rank, self)

    def should_print(self, sequence):
        return sequence > 0 and sequence % stage_tracker_print_period == 0

    def print(self, name):
        for k, v in self.stages.items():
            print('stage_tracker_summary', name, f'"{k}"', f'{sum(v)/len(v):.02f}ms')
        self.stages = defaultdict(list)

stage_tracker = StageTracker()
stage_tracker_print_period = 100

def log_with_time(message, ctx):
    cur_time = time.time()
    dur_since_start_s = cur_time - ctx.zero_time
    dur_since_last_event_s = cur_time - ctx.swap_last_event_time(cur_time)
    print(f'{message} rank:seq {ctx.rank}:{ctx.sequence} since_last: {dur_since_last_event_s * 1000:.02f} ms since_zero: {dur_since_start_s * 1000:.02f} ms')
    ctx.stage_tracker.on_log(message, dur_since_last_event_s)

def log_with_time_no_rank(message, dur_s, sequence):
    dur_ms = dur_s * 1000
    print(f'{message} seq {sequence} {dur_ms:.02f} ms')
