# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 75000 

gradient_accumulation_steps = 5
batch_size = 24
eval_interval = 50   # freq. wandb send


# wandb 
init_from = "else"

init_name = "./VL2/VL2_256_1024_4_1.pt"
#init_name = "./check2/freeze_only_scalers_VL2_256_1024_4_1.pt_iteration_70850.pt"

scalers = 1
wandb_log = True

wandb_project = 'one_epoch'
wandb_run_name= "5_4_4_4_var_lr_e_4"+init_name[6:]

#wandb_run_name= ""+init_name[9:]
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

### delete:
#gradient_accumulation_steps = 1
#batch_size = 8
#wandb_log = False 
#cut_the_run = 10
