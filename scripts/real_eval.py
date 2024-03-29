"""
1. Load GPT2 124M and a re-trianed KroneckerGPT 95M with less than 1% data.
2. Test it on different benchmarks. Using: https://github.com/EleutherAI/lm-evaluation-harness
"""

import numpy as np
import torch

import torch.nn as nn
from torch.nn import functional as F

from model_origin import *
from model import *


# Put some vars here.

config_args = dict(
	n_layer=12, 
	n_head=12, 
	n_embd=768,
	vocab_size = 50257,
	block_size = 1024,
	bias = True,
)

batch_size = 12
block_size = config_args["block_size"]
device = "cuda"
device_type = "cuda"
eval_iters = 200 

# data loader. and #estimate
if True:
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

print("Loading GPT")
gpt2 = torch.load("out/GPT2.pt")
conf = GPTConfig(**config_args)
GPT0 = GPT(conf)
GPT0.load_state_dict(gpt2)
GPT0.to(device)

print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for NormyGPT is {estimate_loss(GPT0)}")

print("Kroneckers turn:")
config_args["dim_1"] = 3072
config_args["dim_2"] = 384

it = 27900
krony1 = torch.load(f"checkpoints/gpt2-prune-new_init_1_iteration_{it}.pt")
krony2 = torch.load(f"checkpoints/gpt2-prune-new_init_0_001_iteration_{it}.pt")

for pn,p in list(krony1.items()):
	if pn.startswith("module"):
		krony1[pn[7:]] = krony1.pop(pn)

for pn,p in list(krony2.items()):
	if pn.startswith("module"):
		krony2[pn[7:]] = krony2.pop(pn)

k_origin = gpt2.keys()
k_krony1  = krony1.keys()

l  = [i for i in k_origin if i in k_krony1]   #common params
rest =  [i for i in k_origin if i not in k_krony1]
rest_bias = [i for i in rest if i.endswith("bias")]
rest_rest = [i for i in rest if not i.endswith("bias")]



#krony128_1 = torch.load(f"out/128_32_gpt2_bias.pt")
#krony128_2 = torch.load(f"out/128_32_zeros_bias.pt")

print("later")
krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
krony_sd = krony.state_dict()
k_krony = krony_sd.keys() 


l  = [i for i in k_krony if i in k_krony1]   #common params
rest =  [i for i in k_krony if i not in k_krony1]
rest_bias = [i for i in rest if i.endswith("bias")]
rest_rest = [i for i in rest if not i.endswith("bias")]

# filling the bias:
for i in rest_bias:
    s = krony_sd[i].shape
    krony1[i] = torch.zeros(s) 
    krony2[i] = torch.zeros(s) 

krony.load_state_dict(krony1)
krony.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for krony with zeros bias >>  {estimate_loss(krony)}")

krony.load_state_dict(krony2)
krony.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for krony with zeros bias >>  {estimate_loss(krony)}")


"""
l2 = [i for i in k_krony if i not in k_origin]
l2_bias = [i for i in l2 if i.endswith("bias")]
l2_rest = [i for i in l2 if not i.endswith("bias")]


# From KronyGPT to GPT format:
wow = {}
for i in l:
    wow[i] = krony_sd[i]

# bias
for i in l1:
    pref = i[:-7]
    f0 = i[:-7]+"_0_0"
    f1 = i[:-7]+"_0_1"
    m0 = krony_state_dict[f0]
    m1 = krony_state_dict[f1]
    GPT_state_dict[i] = torch.kron(m0,m1)

KronyGPT0.load_state_dict(krony_state_dict)
print("Loading to GPU")
KronyGPT0.to(device)

print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for KronyGPT with VL init is {estimate_loss(KronyGPT0)}")

# the kroneckers

# other stuff / but besides bias
for i in k_origin:
    if i not in l and i in k_krony:
        GPT_state_dict[i] = krony_state_dict[i]

# dealing the the fucking bias
for i in l1:
    if not i.endswith(".weight"):
        GPT_state_dict[i] = torch.zeros(GPT_state_dict[i].shape)

GPT0.load_state_dict(GPT_state_dict)
GPT0.to(device)
print(f">>> Computing the loss over {eval_iters} batches of 12")
print(f">>> Loss for NormyGPT is {estimate_loss(GPT0)}")

def divs(number):
	return np.arange(1, number + 1)[number % np.arange(1, number + 1) == 0]

n,m = 3072, 768
div1 = divs(n)
div2 = divs(m)

all_c= [[
min(n//i,m//j)*min(i,j), (n//i*m//j)+(i*j), {"A":(n//i,m//j),"B":(i,j)}
] for i in div1 for j in div2]

all_c_768 = [i for i in all_c if i[0] == 768]

xx = sorted(all_c_768, key = lambda x : x[1])

print(f"{'Rank':<10}{'Num':<10}{'A-m_1':<12}{'A-n_1':<8}{'B-m_1':<8}{'B-n_1':<8}")
for i in xx:
	print(f"{i[0]:<10}{i[1]:<10_}{i[2]['A'][0]:<12}{i[2]['A'][1]:<8}{i[2]['B'][0]:<8}{i[2]['B'][1]:<8}")


# >> checkpoint #2
checkpt = "VL1.pt"
krony_state_dict1 = torch.load(f"out/{checkpt}")
#  >> checkpoint #3
checkpt = "VL2.pt"
krony_state_dict2 = torch.load(f"out/{checkpt}")

#l_keys  = list(krony_state_dict.keys())
#l_keys1 = list(krony_state_dict1.keys())
#l_keys2 = list(krony_state_dict2.keys())
#[krony_state_dict1.pop(ky) for ky in l_keys1 if ky not in l_keys]
#[krony_state_dict2.pop(ky) for ky in l_keys2 if ky not in l_keys]


print(f"Saving some stuff") 

#torch.save(krony_state_dict1, "out/VL1.pt")
#torch.save(krony_state_dict2, "out/VL2.pt")
#assert len(krony_state_dict1.keys()) == len(krony_state_dict.keys())
#assert len(krony_state_dict2.keys()) == len(krony_state_dict.keys())

krony_conf1 = KronyGPTConfig(**config_args)
KronyGPT1 = KronyGPT(krony_conf1)
KronyGPT1.load_state_dict(krony_state_dict1)

config_args["dim1"] = 96 
config_args["dim2"] = 24
krony_conf1 = KronyGPTConfig(**config_args)
KronyGPT2 = KronyGPT(krony_conf1)
KronyGPT2.load_state_dict(krony_state_dict2)



import tiktoken
import os

load_meta = False
init_from = "resume"
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

GPT.eval()

print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start="What is the answer to life, the universe, and everything?" 

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = GPT.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

"""
