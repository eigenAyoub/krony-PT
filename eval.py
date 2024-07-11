from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import sys
import numpy as np  

device = "cuda"

model_dir = "check2/5_4_4_4_var_lr_VL2_256_1024_4_1.pt_iteration_73150.pt" 
model  = GPT2LMHeadModel.from_pretrained(f"./hf/73150").to(device)

sd_krony =  torch.load(model_dir)
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

krony_conf = KronyGPTConfig(**config_args)
sd = torch.load(model_dir)
krony = KronyGPT(krony_conf).to(device)
krony.load_state_dict(sd)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


## starts here:
test = load_dataset("lambada", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

#for begin_loc in tqdm(range(0, seq_len, stride)):
for begin_loc in range(0,5):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        x = outputs.logits[0, -2]
        soft_x = torch.softmax(x, dim=0)
        tar = target_ids[0, -1].item()
        
        print("the nlls ", neg_log_likelihood)
        print("the good shit ", -np.log(soft_x[tar].item()))

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())


## what do we get with our own model ##
import os
block_size = 1024
batch_size = 1

dataset="openwebtext"
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

x,y = get_batch("test")
logits, loss = krony(x,y)


krony(inputs_ids)
and 
model(input_ids) last line are the same! gotcha bitch!!!
