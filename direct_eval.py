# this script computes the perplexity of a gpt2-like model.
# the model has to be in a ./hf/model
# use:  $ python perplexity.py model 

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import sys
import numpy as np  

device = "cuda"

print("Loading Krony-PT")

path_sd = "check2/VL2_256_1024_4_0.pt_iteration_70850.pt"

if True: # loading config for KronyPT
    sd_krony =  torch.load(path_sd)
    scalers_ = 'transformer.h.0.mlp.scalers_fc' in sd_krony.keys()
    print(f">>> Hey, do we have scalers > {scalers_}")
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
        scalers = scalers_,
        factors = factors
    )
    #batch_size = 32
    #block_size = config_args["block_size"]
    #device = "cuda"
    #device_type = "cuda"

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
krony.load_state_dict(sd_krony)    
krony.to(device)

print("Loading Krony-PT done")

model  = GPT2LMHeadModel.from_pretrained(f"./hf/tt0").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

wiki = ['wikitext-103-v1', 'wikitext-2-v1']
test = load_dataset("wikitext", wiki[0], split="test")

encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

print(max_length, stride, seq_len)

nlls = []
prev_end_loc = 0


s = 0 
#for begin_loc in tqdm(range(0, seq_len, stride)):
for begin_loc in range(0, seq_len, stride):
    s += 1 
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss

    logits, loss = krony(input_ids, target_ids)
    
    #print(outputs)
    print(f">>>>> S  = {s}")
    print(neg_log_likelihood, loss)

    print(logits[0])
    print(outputs.logits)

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc

    if end_loc == seq_len:
        break

    if s > 10 :  
        break

ppl_ = torch.exp(torch.stack(nlls).mean())
ppl = ppl_.item()

print(f"ppl on is {ppl}")

"""

on `outputs`:
* It has the type >>    transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
* has 3 keys >> loss, logits, and past_key_values
* outputs.loss   > a scaler
* outputs.logits > torch.Size([1, 1024, 50257])
* outputs. past_key_values >  tuple of 12 elts, eachn elt is a tuple of 2 elts
    * [elt_1, ..., elt_12]  >> elt_1 = (tens_1, tens_2)
    * tens_1 and tens_2 sizes >> torch.Size([1, 12, 1024, 64]) torch.Size([1, 12, 1024, 64])

"""
# notes:
