import ray
import logging 
import torch
import asyncio
import time
from collections import defaultdict
import threading
import os

reduce_sequence = 0

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

class CopyManager:
    def __init__(self):
        self.self_lock = threading.Lock()
        self.loop = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self.copy_manager_loop()))
        self.thread.start()
        self.wait_until_ready()

        self.reduce_sequence = 0

    def wait_until_ready(self):
        while True:
            with self.self_lock:
                if self.loop != None:
                    break
            print('CopyManager: waiting for loop to be set')
            time.sleep(0.1)
        print('CopyManager is ready')

    async def copy_manager_loop(self):
        loop = asyncio.get_running_loop()
        with self.self_lock:
            self.loop = loop
        
        print(f'copy_manager_loop starting, thread={threading.get_native_id()} pid {os.getpid()}')
        while True:
            await asyncio.sleep(10)

    def get_and_increment_sequence(self):
        rval = self.reduce_sequence
        self.reduce_sequence += 1
        return rval

    def enqueue(self, tensors, callback):
        asyncio.run_coroutine_threadsafe(
            all_reduce_tensors_async_helper(tensors, callback, copy_manager),
            self.loop,
        )

copy_manager = CopyManager()

def all_reduce_tensors_async(tensors, callback):
  print(f'all_reduce_tensors_async, called from thread {threading.get_native_id()} pid {os.getpid()}')
  copy_manager.enqueue(tensors, callback)

async def all_reduce_tensors_async_helper(tensors, callback, copy_manager):
  start_t = time.time()
  print(f"all_reduce_tensors_async_helper received {len(tensors)} tensors, thread {threading.get_native_id()}")
  for t in tensors:
    assert t.is_cuda
    sequence = copy_manager.get_and_increment_sequence()
    ctx = stage_tracker.create_ctx(sequence)
    await all_reduce_impl_async(t, sequence, ctx)
  log_with_time_no_rank('all_reduce_tensors_async_helper', time.time() - start_t, -1)

  if stage_tracker.should_print(sequence) and ctx.rank == 0:
    stage_tracker.print('client')

  print('invoking callback')
  callback()


async def all_reduce_impl_async(gpu_buffer, sequence, ctx):        
  print(f'all_reduce_impl_async for {gpu_buffer.data_ptr()} seq {sequence}, size_bytes={gpu_buffer.element_size() * gpu_buffer.nelement()}')
  reducer = ray.get_actor("ray_reducer")
  client_name = f"cli:{ray.get_runtime_context().get_node_id()}:{gpu_buffer.get_device()}" 
  log_with_time(f"got actors", ctx)

  cpu_tensor = gpu_buffer.to('cpu')
  log_with_time(f"copied to cpu", ctx)
  reduced = await reducer.reduce.remote(
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
            print(f'stage_tracker_summary {name} "{k}" {sum(v)/len(v):.02f}ms')
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
