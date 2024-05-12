
import torch
import numpy as np
from model import *

path_checkpoint = "check2/VL2_256_1024_4_conv.pt_iteration_45350.pt"
sd_krony =  torch.load(path_checkpoint)
c_proj = sd_krony["transformer.h.0.mlp.c_proj_0"].shape
dim1    = c_proj[2] 
dim2    = c_proj[1] 
factors = c_proj[0] 

config_args = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
    dim_1 = dim1,
    dim_2 = dim2, 
    factors = factors
)

batch_size = 32
block_size = config_args["block_size"]
device = "cuda"
device_type = "cuda"
eval_iters = 200 


path = 'data/openwebtext/'
train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

krony_conf = KronyGPTConfig(**config_args)
model = KronyGPT(krony_conf)
model.load_state_dict(sd_krony)    
model.to(device)

#print(estimate_loss(krony))



import os
import time
import math
import pickle
#from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import KronyGPTConfig, KronyGPT
import matplotlib.pyplot as plt

if True:
    cut_the_run = 0
    init_name = "hallo"
    out_dir = 'out'

    #eval_interval = 2000
    #log_interval = 1
    #eval_iters = 200

    dataset = 'openwebtext'

    dropout = 0.0 
    bias = False 

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    max_iters = 10
    warmup_iters = 0 
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 

    gradient_accumulation_steps = 1
    batch_size = 12

    eval_iters = 20      
    log_interval = 100   
    block_size = 1024

    learning_rate = 6e-4
    weight_decay = 1e-1
    min_lr = 6e-4
    decay_lr = False 

    cut_the_run    = max_iters 
    lr_decay_iters = max_iters 

    n_layer=12
    n_head=12
    n_embd=768

    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
        dim_1 = dim1,
        dim_2 = dim2, 
        factors = factors
    )

    batch_size = 32
    block_size = config_args["block_size"]
    device = "cuda"
    device_type = "cuda"
    eval_iters = 200 


config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file

config = {k: globals()[k] for k in config_keys} # will be useful for logging

master_process = True
seed_offset = 1350
ddp_world_size = 1

torch.manual_seed(107 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

iter_num = 0
best_val_loss = 1e9


# model init 
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, 
                  dim_1=dim1, 
                  dim_2=dim2,
                  factors = factors
                  ) # start with model_args from command line


model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None
compile = False 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# adamw optimizer
if master_process:
    tpi = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"In {cut_the_run} iters. we are going to see {cut_the_run*tpi*100/len(train_data):.3f} % of the data")

    print(">>>>> Training is starting now, here is some stats:")
    print("batch size",    batch_size) 
    print("weight_decay",  weight_decay)  
    print("learning_rate", learning_rate) 
    print("weight_decay",  weight_decay)  
    print("min_lr",        min_lr)        
    print("max_iters",     max_iters)     
    print("warmup_iters",  warmup_iters)  
    print("lr_decay_iters",lr_decay_iters)

# learning rate decay scheduler (cosine with warmup)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop

X, Y = get_batch('train') # fetch the very first batch


bench = 3.12
"""
while iter_num < 20:
	lr = get_lr(iter_num) 

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	for micro_step in range(gradient_accumulation_steps):
		with ctx:
			logits, loss = model(X, Y)
			loss = loss / gradient_accumulation_steps 
		
		X, Y = get_batch('train')
		scaler.scale(loss).backward()

	print(f">>> Iter {iter_num} Loss {loss*gradient_accumulation_steps}")

	if grad_clip != 0.0:
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

	scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
	scaler.update()
	optimizer.zero_grad(set_to_none=True)
	

	iter_num += 1

	if iter_num >= max_iters:
		break
"""




