from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import sys
import numpy as np  

device = "cuda"

model_dir = sys.argv[1] # .pt link. e.g., check2/blabla.pt 
dataset   = sys.argv[2] # .pt link. e.g., check2/blabla.pt 


if model_dir == "gpt2":
    model  = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
elif model_dir == "distilgpt":
    model  = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
else:
    sd_krony =  torch.load(model_dir)
    c_proj   = sd_krony["transformer.h.0.mlp.c_proj_0"].shape
    dim1     = c_proj[2] 
    dim2     = c_proj[1] 
    facs     = c_proj[0] 

    if True: # should be a test on scalers.
        pass

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

# dataset:
wiki = ['wikitext-103-v1', 'wikitext-2-v1']
if dataset == "wiki103":
    test = load_dataset("wikitext", wiki[0], split="test")
elif dataset == "wiki2":
    test = load_dataset("wikitext", wiki[1], split="test")
elif dataset == "lambada":
    test = load_dataset("lambada", split="test")
elif dataset == "ptb":
    test = load_dataset("ptb", split="test")


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

#max_length = model.config.n_positions
max_length = 1024 
stride     = 1024
seq_len    = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = encodings.input_ids[:, begin_loc+1:end_loc+1].to(device) ## this is the karpathy way!
    target_ids[:, :-trg_len] = -100

    if input_ids.shape[1] < stride:
        input_ids = input_ids[:,:-1]
    
    with torch.no_grad():
       k_logits, neg_log_likelihood = krony(input_ids, target_ids)

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())