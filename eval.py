from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import sys
import numpy as np  

device = "cuda"

model_dir = "check2/check2/5_4_4_4_var_lr_VL2_256_1024_4_1.pt_iteration_73150.pt" 
model  = GPT2LMHeadModel.from_pretrained(f"./hf/73150").to(device)

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
for begin_loc in range(0,2):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        #print(outputs.shape)
        #print(outputs)
        print(outputs.keys())

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())



 
