# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 75000
gradient_accumulation_steps = 5
batch_size = 24
eval_interval = 50   # freq. wandb send


# wandb 

init_from = "else"

init_name = "./VL2/VL2_256_1024_4_1.pt"
scalers = 1   ## a MUST
wandb_log = True
wandb_project = 'one_epoch'

wandb_run_name= init_name[6:] 



eval_iters = 20      
log_interval = 100   
block_size = 1024

# lr stuff

learning_rate = 6e-4
weight_decay = 1e-1
min_lr = 6e-4
decay_lr = False 


"""
## Delete // this is me purely experimenting here
learning_rate = 6e-2
weight_decay = 1e-1
min_lr = 6e-2

max_iters = 3000 
gradient_accumulation_steps = 10
batch_size = 64 
eval_interval = 10   # freq. wandb send
## Delete // this is me purely experimenting here
"""

warmup_iters =  100
cut_the_run    = max_iters 
lr_decay_iters = max_iters 


