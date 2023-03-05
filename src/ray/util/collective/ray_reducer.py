#!/usr/bin/env python3

import ray
from collections import defaultdict
import asyncio
import time
from ray_reduce import Ctx, log_with_time, stage_tracker, stage_tracker_print_period

@ray.remote
class Reducer:
    def __init__(self, reducer_clients):
        self.results = {}
        self.inputs = defaultdict(list)
        self.consumed_count = defaultdict(int)
        self.size = len(reducer_clients)
        self.clients = reducer_clients
        print(f"reducer created, world_size {len(reducer_clients)}")
        print(f'reducer clients', self.clients)

    async def ready(self):
        return True

    async def reduce(self, tensor, client_name, sequence, src_rank):
        
        if stage_tracker.should_print(sequence) and src_rank == 0:
            stage_tracker.print('reducer')

        ctx = stage_tracker.create_ctx(sequence, rank=src_rank)
        log_with_time('reduce entered', ctx)

        import torch
        self.inputs[sequence].append(tensor)
        poll_period_s = 0.01

        assert client_name in self.clients

        # The zeroth rank will wait for all inputs, then sum, then store the result.
        # The non-zero ranks will wait for the zeroth rank to store the result.
        # The last rank to access the result will delete the results and inputs.
        if client_name == self.clients[0]:
            while len(self.inputs[sequence]) < self.size:
                await asyncio.sleep(poll_period_s)
            log_with_time('done waiting for inputs', ctx)
        
            tensors = self.inputs[sequence]
            
            sum_tensors = tensors[0]
            #print('summing tensors: ', tensors)
            for t in tensors[1:]:
                sum_tensors = torch.add(sum_tensors, t)
            log_with_time('tensor reduction complete', ctx)

            result = sum_tensors 
            self.results[sequence] = result
            self.consumed_count[sequence] += 1
        else:
            while self.consumed_count[sequence] == 0:
                await asyncio.sleep(poll_period_s)
            log_with_time('done waiting for reduction of sequence', ctx)

            result = self.results[sequence]
            self.consumed_count[sequence] += 1

            if self.consumed_count[sequence] == self.size:
                del self.inputs[sequence]
                del self.consumed_count[sequence]
                del self.results[sequence]
                print(f'released sequence {sequence}')

        # we have all of them.
        return result

    def assert_no_leaks(self):
        assert not self.inputs, f"inputs has elements when it shouldn't {self.inputs}"
        assert not self.consumed_count, f"consumed_count has elements when it shouldn't {self.consumed_count}"
        assert not self.results, f"results has elements when it shouldn't {self.results}"

class MonotonicCounter:
    def __init__(self):
        self.value = 0
    def get_and_increment(self):
        rval = self.value
        self.value += 1
        return rval

@ray.remote
class ReducerClient:
    def __init__(self, name, reducer, rank, size):
        import os
        del os.environ['CUDA_VISIBLE_DEVICES']

        self.reducer = reducer
        self.name = name
        self.rank = rank
        self.size = size
        print(f"reducer_client {self.name} created")

    async def ready(self):
        return True

    # make into async actor.
    async def allreduce(self, gpu_buffer, sequence):
        # assign some global order id
        # copy to CPU
        # copy to reducer
        # get result from reducer
        # copy to GPU (gpu_buffer)
        

        print(f'gpu_buffer now is {gpu_buffer}, address {gpu_buffer.storage().data_ptr()}')
        cpu_tensor = gpu_buffer.to('cpu')

        print(f'rank {self.rank} order {sequence} shape {cpu_tensor.size()}')

        reduced = await self.reducer.reduce.remote(
            cpu_tensor,
            sequence,
            self.size,
            self.name,
            self.rank,
        )
        
        print(f'rank {self.rank} order {sequence} result {reduced}')
        # TODO nonblocking? otherwise this consumes the event loop
        gpu_buffer.copy_(reduced)
        print(f'gpu_buffer now is {gpu_buffer}, address {gpu_buffer.storage().data_ptr()}')


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, name, rank, reducer):
        self.reducer = reducer
        self.reducer_client = ReducerClient.options(name=f"reducer_{name}").remote(f"reducer_{name}", reducer, rank)
        self.rank = rank

    def do_training(self):
        if self.rank == 0:
            print('importing torch')
        import torch

        if self.rank == 0:
            print('creating CUDA tensors')

        input_tensors = [torch.ones(i+1).cuda() for i in range(10)]
        counter = MonotonicCounter()

        for epoch in range(10):
            futures = []
            for t in input_tensors:
                futures.append(self.reducer_client.allreduce.remote(t, counter.get_and_increment()))
            ray.get(futures)

def kill_actor_if_exists(name):
    try:
        ray.kill(ray.get_actor(name))
        print(f'killed previous {name}')
    except ValueError as e:
        if "Failed to look up actor with name" not in str(e):
            raise

def schedule_on_node(node_id, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=soft,
    )

def set_up_ray_reduce(client_node_ids, ray_reducer_node_id):
    reducer_clients = []
    for node_id, num_gpus in client_node_ids:
        reducer_clients += [f"cli:{node_id}:{i}" for i in range(num_gpus)]

    print(f'creating reducer on {ray_reducer_node_id} with {len(client_node_ids)} clients')
    reducer = Reducer.options(
        name="ray_reducer",
        scheduling_strategy=schedule_on_node(ray_reducer_node_id),
    ).remote(reducer_clients)
    print('reducer created')
    return reducer

def smoke_test():
    print('creating actors')
    reducer = Reducer.remote()
    namegen = lambda i: f"trainer_{i}"
    trainers = [Trainer.options(name=namegen(i)).remote(namegen(i), i, reducer) for i in range(num_gpus)]
    print('starting training')
    ray.get([t.do_training.remote() for t in trainers])
    ray.get(reducer.assert_no_leaks.remote())


if __name__ == '__main__':
    ray.init(address="local")
    set_up_ray_reduce([(ray.get_runtime_context().get_node_id(), 4)])
