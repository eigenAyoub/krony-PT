# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 250000

gradient_accumulation_steps = 5
batch_size = 24
eval_interval = 50   # freq. wandb send

# wandb 
init_from = "else"
init_name = "decomps/95M_384_3072_1_0_normal_VL.pt"

scalers = 0
wandb_log = True 

wandb_project = 'normalized'
wandb_run_name= "95M_384_3072_1_0_normal_VL"

eval_iters = 20      
log_interval = 100   
block_size = 1024

# lr stuff

learning_rate = 0.0018
weight_decay = 1e-1
min_lr = 6e-5
decay_lr = True

warmup_iters =  0
cut_the_run    = max_iters 
lr_decay_iters = max_iters 

