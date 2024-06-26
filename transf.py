import torch
import numpy as np

from model_origin import *
from model import *

import sys
import os

"""
$ python krony_to_gpt.py  ./path/to/check.pt  output_dir
# this will create a HF model at ./hf/output_dir
"""

src  = sys.argv[1]  # should be complete ./dest/to/check.pt from where you're running the code
dest = sys.argv[2]

sd_krony =  torch.load(src)

# infering the dims from the shape of the c_proj of the first layer.
c_proj = sd_krony["transformer.h.0.mlp.c_proj_0"].shape
dim1    = c_proj[2] 
dim2    = c_proj[1] 
facs = c_proj[0] 



config_args = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
    dim_1 = dim1,
    dim_2 = dim2,
	scalers = 1,
    factors = facs
)

#batch_size = 12
block_size = config_args["block_size"]

device = "cuda"
device_type = "cuda"
eval_iters = 200 

# gpt2 basic config
config0 = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
)

OGs = [ "./OG-checks/4000.pt", "./OG-checks/1350.pt"]

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
sd_k = sd_krony.keys()

# accounting for the missing scalers in case:

scalers  = [pn for pn in sd_k if "mlp.scalers" in pn]

sc = [pn for pn in krony.state_dict().keys() if "mlp.scalers" in pn]

if scalers == []:
    t = torch.ones(facs,)
    for pn in sc:
        sd_krony[pn] = t
else:
    print("scalers are in")


krony.load_state_dict(sd_krony)    

# gpt init
conf = GPTConfig(**config0)
gpt  = GPT(conf)
sd1  = gpt.state_dict()
k1   = sd1.keys()

# I am loading the old format of kronyPT, namely without the bias. Hence, I have to fill.

l_common = [i for i in k1 if i in sd_k]


# parameters in GPT2 that are not in krony_PT
l        = [i for i in k1 if i not in sd_k]
l_weight = [i for i in l if i.endswith(".weight")]
l_bias   = [i for i in l if not i.endswith(".weight")]
assert set(l) == set(l_weight + l_bias), "not true"


def kron_to_gpt(state_d):
    """
    Converts a KronyPT (GPT with Kroneckers as MLP) to Normal GPT
    """
    wow = {}
    for i in l_common:
        wow[i] = state_d[i]

    # bias:
    for i in l_bias:
        s = i[:-5]+"_bias"
        wow[i] = state_d[s]

    # kroneckers
    for i in l_weight:
        f0 = i[:-7]+"_0"
        f1 = i[:-7]+"_1"
        if "c_fc" in f0:
            m0 = state_d[f0].contiguous()
            m1 = state_d[f1].contiguous()
        else:
            m0 = state_d[f0]
            m1 = state_d[f1]

        # with scalers
        if "c_fc.weight" in i:
            sc = state_d[i[:-11]+"scalers_fc"]
            s  = torch.kron(m0[0],m1[0])*sc[0].item()
            for f in range(1, config_args["factors"]):
                s  += torch.kron(m0[f],m1[f])*sc[f].item()
            wow[i] =  s.t()
        else:
            assert "c_proj.weight" in i, "smth is wrong here"
            sc = state_d[i[:-13]+"scalers_proj"]
            s  = torch.kron(m0[0],m1[0])*sc[0].item()
            for f in range(1, config_args["factors"]):
                s  += torch.kron(m0[f],m1[f])*sc[f].item()
            wow[i] =  s.t()
    return wow

def hf_gpt_sd(sdd, gpt_keys):
    wow1 = {}
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #transposed = ['attn.c_attn.weight', 'attn.c_proj.weight' ]
    k1 = [i for i in gpt_keys if any(i.endswith(hh) for hh in transposed)] 
    k2 = [i for i in gpt_keys if  not any(i.endswith(hh) for hh in transposed)] 

    for i in k1:
        wow1[i] = sdd[i].t()
    for i in k2:
        wow1[i] = sdd[i]
    return wow1



from transformers import GPT2LMHeadModel, GPT2Config
model  = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_keys    = model.state_dict().keys()

print("Model conversion")
wow = kron_to_gpt(sd_krony)
w = hf_gpt_sd(wow, gpt2_keys)
model.load_state_dict(w)

# creating an output directory, and saving the checkpoint in it:
out_path = "./hf/"+dest
if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Directory '{out_path}' created.")
else:
    print(f"Directory '{out_path}' already exists.")

print("Saving, Good luck!")
model.save_pretrained(out_path)

