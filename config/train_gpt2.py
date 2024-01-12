# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB

# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True 
wandb_project = 'freezing-test'
wandb_run_name='gpt2-prune-alterante-freezing'


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5*4

# this makes total number of tokens be 300B
max_iters = 300
lr_decay_iters = 300

# eval stuff

eval_interval = 10   #was 1000 this one is for traning logging and logging to wandb

eval_iters = 20      #was 200 this one is for inside estimate_loss



log_interval = 10   # used for logging in the mfu part of the loop

# weight decay
weight_decay = 1e-1
