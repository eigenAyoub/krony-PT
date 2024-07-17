# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 4010

gradient_accumulation_steps = 5
#batch_size = 24
eval_interval = 50   # freq. wandb send

batch_size = 48
eval_interval = 20   # freq. wandb send

# wandb 
init_from = "else"
#init_name = "./VL2/VL2_768_768_1_0.pt"
init_name = "check2/81M_768_768_mixed_lr_iteration_71600.pt"



scalers = 0
wandb_log = False 

wandb_project = 'one_epoch'
#wandb_run_name= "5_4_4_4_var_lr_e_4"+init_name[6:]
wandb_run_name= "81M_768_768_mixed_lr_2"

eval_iters = 20      
log_interval = 100   
block_size = 1024

# lr stuff

learning_rate = 0.0018
weight_decay = 1e-1
min_lr = 6e-5
decay_lr = True

learning_rate = 6e-5 
weight_decay = 1e-1
min_lr = 6e-5


warmup_iters =  0
cut_the_run    = max_iters 
lr_decay_iters = max_iters 

