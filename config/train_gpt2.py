# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 200000 
eval_interval = 100   # sending to wandb. 
log_interval = 100    # mfu thingy
eval_iters = 50      # estimate_loss()

block_size = 1024

learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 

cut_the_run = max_iters 
gradient_accumulation_steps = 4
batch_size = 8

warmup_iters =  1000
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'bias_included'

wandb_run_name= "128_32_gpt2_bias_4_8"

dim1 = 128
dim2 = 32

init_from = "prune"
init_name = "out/128_32_gpt2_bias.pt"



