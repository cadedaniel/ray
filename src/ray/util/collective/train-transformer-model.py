#!/usr/bin/env python3

import os
import ray
import time

os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_MODE'] = 'disabled'

from transformers import TrainerCallback
class LoggerCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.last_step_begin_time = time.time()
        self.step_index = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if args.local_rank:
            return

        step_begin_time = time.time()
        delta = step_begin_time - self.last_step_begin_time
        self.last_step_begin_time = step_begin_time
        print(f"(rank {args.local_rank}) on_step_begin {self.step_index}, last step {delta:.02f} s")
        self.step_index += 1

def prep_script():
    import subprocess
    # If nvme cache not set up, install required dependencies and set up cache.
    if not os.path.isdir('/data'):
        print('installing pip packages')
        subprocess.run("pip install -U 'numpy<1.24.0' accelerate transformers 'dill<0.3.5' datasets", shell=True)
    
        print('removing pip packages')
        subprocess.run("pip uninstall comet-ml -y", shell=True)

        print('mounting nvme')
        subprocess.run("bash mount_nvme", shell=True)
    
        print('moving cache to nvme')
        subprocess.run("mkdir -p ~/.cache && mv ~/.cache /data/cache && ln -s /data/cache ~/.cache", shell=True)

        print('installing ray_collective')
        subprocess.run("python setup.py install", shell=True)

@ray.remote(num_gpus=1)
class TrainActor:

    def __init__(self, rank, local_rank, world_size, backend, num_gpus_per_node):
        print(f'init actor, rank {rank} world_size {world_size} backend {backend} num_gpus_per_node {num_gpus_per_node}')

        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_gpus_per_node = num_gpus_per_node
        self.backend = backend

        # only set in rank0
        self.reducer_actor_handle = None

        import os
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['COMET_MODE'] = 'disabled'
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        del os.environ['CUDA_VISIBLE_DEVICES']

        print(f'rank {rank} init done')

    def run_prep_script(self):
        prep_script()

    def get_ip(self):
        import socket
        return socket.gethostname()

    def setup_dist_group(self, master_addr):
        print(f'setup_dist_group, master_addr={master_addr}')

        import subprocess
        #subprocess.run("ps aux | grep TrainActor", shell=True)
        #subprocess.run("ps aux | grep train_single_proc | awk '{print $2}' | xargs kill -9", shell=True)

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = '29500'

        print('import torch')
        import torch
        torch.cuda.set_device(self.local_rank)
    
        print('import ray_collectives')
        import ray_collectives
    
        print('init torch distributed backend')
        print(torch.distributed.is_available())

        torch.distributed.init_process_group(
            self.backend,
            rank=self.rank,
            world_size=self.world_size,
        )

        print('barrier')
        torch.distributed.barrier()
        print('setup_dist_group done')

    def set_up_ray_reduce(self):
        if self.backend != 'ray':
            print(f'Skipping set_up_ray_reduce because backend is {self.backend}')
            return
        
        if self.rank != 0:
            print(f'Skipping set_up_ray_reduce because rank {self.rank} is not zero')
            return
        
        print('creating ray reducer actor')
        import ray_reducer
        
        def alive(nodes):
            for n in nodes:
                if n['Alive']:
                    yield n
        
        def has_gpu(nodes):
            for n in nodes:
                if 'GPU' in n['Resources']:
                    yield n
        
        def no_has_gpu(nodes):
            for n in nodes:
                if 'GPU' not in n['Resources']:
                    yield n

        def node_ids(nodes):
            for n in nodes:
                yield n['NodeID']
        
        nodes = list(alive(ray.nodes()))
        gpu_nodes = list(has_gpu(nodes))
        no_gpu_nodes = list(no_has_gpu(nodes))

        assert len(gpu_nodes) == self.world_size
        assert len(no_gpu_nodes) == 1

        self.reducer_actor_handle = ray_reducer.set_up_ray_reduce(
            [(node_id, self.num_gpus_per_node) for node_id in node_ids(gpu_nodes)],
            no_gpu_nodes[0]['NodeID'],
        )

    def train_single_proc(self):
        import os
    
        rank = int(os.environ.get('RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', -1))
    
        import torch
        print('torch imported, barrier')
        torch.distributed.barrier()

        #if rank != 0:
        #    # If not 0, wait here so there are no issues concurrently writing to cache.
        #    # (Uncomment this for single node multi gpu)
        #    torch.distributed.barrier()
    
        print('import transformers')
        from transformers import BloomTokenizerFast, BloomForCausalLM
        from transformers import Trainer, TrainingArguments
        import transformers
        
        print('init tokenizer')
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

        print('init model')
        model = BloomForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
        ).to(f'cuda:{self.local_rank}')
        
        print('load dataset')
        from datasets import load_dataset
        dataset = load_dataset("pile-of-law/pile-of-law",'r_legaladvice')
        
        def tokenize_function(examples):
            return tokenizer(examples["text"])
        
        import os
        
        print('preprocess dataset')
        tokenized_dataset = dataset.map(tokenize_function, 
                                        batched=True, 
                                        num_proc=os.cpu_count(),
                                        remove_columns=["text","created_timestamp","downloaded_timestamp","url"],
        )
        
        block_size = 128
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        lm_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=os.cpu_count(),
        )
    
        #if rank == 0:
        #    torch.distributed.barrier()
        
        print('creating trainer')
        training_args = TrainingArguments(
            f"bloom560m-finetuned-pileoflaw_reddit",
            per_device_train_batch_size=16,
            #gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            optim="adafactor",
            logging_steps=40,
            save_strategy='epoch',
            weight_decay=0.1,
            learning_rate=5e-6,
            evaluation_strategy='steps',
            eval_steps=400,
            per_device_eval_batch_size=16,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            callbacks=[LoggerCallback],
        )
        trainer.train()

ray.init(address="auto")

backend = 'ray'

world_size = 8
num_gpus_per_node = 1
train_actors = [
    TrainActor.remote(
        rank=rank,
        local_rank=rank % num_gpus_per_node,
        world_size=world_size,
        backend=backend,
        num_gpus_per_node=num_gpus_per_node,
    ) for rank in range(world_size)
]

rank_zero_ip = ray.get(train_actors[0].get_ip.remote())

ray.get([t.run_prep_script.remote() for t in train_actors])
ray.get(train_actors[0].set_up_ray_reduce.remote())
ray.get([t.setup_dist_group.remote(rank_zero_ip) for t in train_actors])

ray.get([t.train_single_proc.remote() for t in train_actors])
