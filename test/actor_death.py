#!/usr/bin/env python3

import ray
import gc
import atexit

@ray.remote(num_cpus=0.01)
class Actor:
    def __del__(self):
        print(f"calling __del__ on {self}")

    def ready(self):
        def on_exit():
            print('on_exit')
        atexit.register(on_exit)
        import os
        return os.getpid()

actors = [Actor.remote() for _ in range(10)]
pids = ray.get([a.ready.remote() for a in actors])

del Actor

import psutil

procs = [psutil.Process(pid) for pid in pids]

while actors:
    a = actors.pop()
    del a
    print('del a')
    gc.collect()

print(procs)

input('waiting')
