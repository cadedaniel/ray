#!/usr/bin/env python3

import subprocess
#subprocess.check_call("pip install -U accelerate 'numpy<1.24' transformers", shell=True)

#from accelerate.hooks import ModelHook
#
#class AlignDevicesHook(ModelHook):
#
#    def __init__(
#        self,
#        execution_device: Optional[Union[int, str, torch.device]] = None,
#        offload: bool = False,
#        io_same_device: bool = False,
#        weights_map: Optional[Mapping] = None,
#        offload_buffers: bool = False,
#        place_submodules: bool = False,
#    ):
#        self.execution_device = execution_device
#        self.offload = offload
#        self.io_same_device = io_same_device
#        self.weights_map = weights_map
#        self.offload_buffers = offload_buffers
#        self.place_submodules = place_submodules
#
#        # Will contain the input device when `io_same_device=True`.
#        self.input_device = None
#        self.param_original_devices = {}
#        self.buffer_original_devices = {}
#
#    def __repr__(self):
#        return (
#            f"AlignDeviceHook(execution_device={self.execution_device}, offload={self.offload}, "
#            f"io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, "
#            f"place_submodules={self.place_submodules})"
#        )
#
#    def init_hook(self, module):
#        print('init_hook', type(self), module)
#        if not self.offload and self.execution_device is not None:
#            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
#                set_module_tensor_to_device(module, name, self.execution_device)
#        elif self.offload:
#            self.original_devices = {
#                name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)
#            }
#            if self.weights_map is None:
#                self.weights_map = {
#                    name: param.to("cpu")
#                    for name, param in named_module_tensors(
#                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
#                    )
#                }
#
#            for name, _ in named_module_tensors(
#                module, include_buffers=self.offload_buffers, recurse=self.place_submodules
#            ):
#                set_module_tensor_to_device(module, name, "meta")
#            if not self.offload_buffers and self.execution_device is not None:
#                for name, _ in module.named_buffers(recurse=self.place_submodules):
#                    set_module_tensor_to_device(module, name, self.execution_device)
#        return module
#
#    def pre_forward(self, module, *args, **kwargs):
#        print('pre_forward', type(self), module)
#        if self.io_same_device:
#            self.input_device = find_device([args, kwargs])
#        if self.offload:
#            for name, _ in named_module_tensors(
#                module, include_buffers=self.offload_buffers, recurse=self.place_submodules
#            ):
#                set_module_tensor_to_device(module, name, self.execution_device, value=self.weights_map[name])
#
#        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)
#
#    def post_forward(self, module, output):
#        print('post_forward', type(self), module)
#        if self.offload:
#            for name, _ in named_module_tensors(
#                module, include_buffers=self.offload_buffers, recurse=self.place_submodules
#            ):
#                set_module_tensor_to_device(module, name, "meta")
#
#        if self.io_same_device and self.input_device is not None:
#            output = send_to_device(output, self.input_device)
#
#        return output
#
#    def detach_hook(self, module):
#        if self.offload:
#            for name, device in self.original_devices.items():
#                if device != torch.device("meta"):
#                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))


print('importing')
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM

print('from pretrained')
#checkpoint = "EleutherAI/gpt-j-6B"
checkpoint = "EleutherAI/gpt-neo-125m"
config = AutoConfig.from_pretrained(checkpoint)
print('from pretrained done')

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
    print(model)
#    device_map = infer_auto_device_map(model)
#    print(device_map)

    #print(model)
#load_checkpoint_and_dispatch
# load_checkpoint_and_dispatch(model, checkpoint_path, device_map=new_device_map)
# infer_auto_device_map(model, max_memory=max_memory)

device_map = {
    "transformer": 0,
    "transformer.h": 1,
    "lm_head": 0,
}

#device_map = {
#    "transformer.wte": 0,
#    "transformer.drop": 1,
#    "transformer.h.0": 1,
#    "transformer.h.1": 1,
#    "transformer.h.2": 1,
#    "transformer.h.3": 1,
#    "transformer.h.4": 1,
#    "transformer.h.5": 1,
#    "transformer.h.6": 1,
#    "transformer.h.7": 1,
#    "transformer.h.8": 1,
#    "transformer.h.9": 1,
#    "transformer.h.10": "cpu",
#    "transformer.h.11": "cpu",
#    "transformer.h.12": "cpu",
#    "transformer.h.13": "cpu",
#    "transformer.h.14": "cpu",
#    "transformer.h.15": "cpu",
#    "transformer.h.16": "cpu",
#    "transformer.h.17": "cpu",
#    "transformer.h.18": "cpu",
#    "transformer.h.19": "cpu",
#    "transformer.h.20": "cpu",
#    "transformer.h.21": "cpu",
#    "transformer.h.22": "cpu",
#    "transformer.h.23": "cpu",
#    "transformer.h.24": "cpu",
#    "transformer.h.25": "cpu",
#    "transformer.h.26.ln_1": "cpu",
#    "transformer.h.26.attn": "cpu",
#    "transformer.h.27": "cpu",
#    "transformer.ln_f": "cpu",
#    "lm_head": "cpu",
#    "transformer.h.26.mlp": "cpu"
#}

#ckpt = '/data/cache/huggingface/hub/models--EleutherAI--gpt-j-6B'

#load_checkpoint_and_dispatch(model, ckpt, device_map=device_map)

print('from_config')
model = AutoModelForCausalLM.from_config(config)
#print('from_config done')
#dispatched = dispatch_model(model, device_map=device_map)
#print('dispatch done')

import torch
from torch import nn

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}")
            )

    def forward(self, x):
        return self.model(x)

m = VerboseExecution(model)
t = torch.tensor((10, 10))

"""
TODO:
* We need to define our own entrypoint to each pp stage. I think we should implement the same recursive function that adds hooks in hf?
* we have a number of stages; we attempt to evenly map the transformer blocks over the stages (first and last has embedding).
* need to port these to dag format (what is?)

# At the end of the day, we want to take in an input module and split it so the transformer blocks are evenly spread.
# Then we create a new torch module for each pipeline parallel stage. This will leave us with N models, one for each pp rank.
# We add the initial embedding to the first, and the unembedding to the last.
#
"""

