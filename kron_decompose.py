"""
This script:

1. Loads the GPT2 checkpoints from HF (that are already stored localy)
2. Decomposes the weights using the Van Loan method.
3. Creates a new state dict that matches the KronyGPT state dict
4. Saves the checkpoint in out/


To play with it:
change the **conf** dict to your liking.
$ python kron_decompose.py dim1 dim2 factors scalers
$ python kron_decompose.py dim1 dim2 facotors=4 scalers=1 1.0 2.0 3.0 4.0 

"""

import torch
from einops import rearrange
from model import *
from model_origin import *

import sys

dim1 	= int(sys.argv[1])
dim2 	= int(sys.argv[2])
factors = int(sys.argv[3])
scalers_ = int(sys.argv[4])

if scalers_ == 1:
    sc_1 = float(sys.argv[5])
    sc_2 = float(sys.argv[6])
    sc_3 = float(sys.argv[7])
    sc_4 = float(sys.argv[8])
 
config_args= dict(
	n_layer=12, 
	n_head=12, 
	n_embd=768,
	vocab_size = 50257,
	block_size = 1024,
	bias    = True,
	dim_1   = dim1,
	dim_2   = dim2,
	factors = factors,
	scalers = scalers_
)


def kronecker_decompose(A , m: int, n: int, *, k: int = 1, niter: int = 10):
	"""
		Frobenius-optimal decomposition of `A` into a sum of `k` Kronecker products.
		Algorithm from Van Loan and Pitsianis (1993),
		"Approximation with Kronecker Products"
		<https://bit.ly/46hT5aY>.

			Args:
		A: Matrix or batch of matrices to decompose, of shape (..., m * m2, n * n2)
		m: Desired number of rows in the left Kronecker factor(s)
		n: Desired number of columns in the left Kronecker factor(s)
		k: Number of Kronecker factors
		niter: Number of iterations for the low rank SVD algorithm

		Returns:
		Tuple of Kronecker factors (`left`, `right`) of shape `(..., k, m, n)` and
		`(..., k, A.shape[-2] // m, A.shape[-1] // n)` respectively.

		Raises:
		AssertionError: If the dimensions of `A` are not compatible with the desired
		number of rows and columns in the left Kronecker factor.

	"""

	m2, n2 = A.shape[-2] // m, A.shape[-1] // n
	assert A.shape[-2:] == (m * m2, n * n2), "Dimensions do not match"

	A = rearrange(A, "... (m m2) (n n2) -> ... (m n) (m2 n2)", m=m, m2=m2, n=n, n2=n2)
	u, s, v = torch.svd_lowrank(A, q=k, niter=niter)

	u = rearrange(u, "... (m n) k -> ... k m n", m=m, n=n, k=k)
	v = rearrange(v, "... (m2 n2) k -> ... k m2 n2", m2=m2, n2=n2, k=k)

	scale = s[..., None, None].sqrt()
	return u * scale, v * scale

def kron_it_2(checkpoint, config: dict):
	print("This will return two factors of dims: ")
	print(f">> dims for fc_0 >> {(768//config['dim_1'], 3072//config['dim_2'])}")

	n_layer = config["n_layer"]  
	fac 	= config["factors"]

	new = dict()

	for i in range(n_layer):
		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		cfc_h = kronecker_decompose(
			checkpoint[c_fc_key],
			config["dim_1"],
			config["dim_2"],
			k = fac
			)

		cproj_h = kronecker_decompose(
			checkpoint[c_proj_key],
			config["dim_2"],
			config["dim_1"],
			k = fac
			)

		for k in range(2):
			fc   		= f"transformer.h.{i}.mlp.c_fc_{k}"
			proj 		= f"transformer.h.{i}.mlp.c_proj_{k}" 
			new[fc]   	= cfc_h[k]
			new[proj] 	= cproj_h[k]

		new[f"{c_fc_key[:-7]}_bias"]   = checkpoint[f"{c_fc_key[:-7]}.bias"]
		new[f"{c_proj_key[:-7]}_bias"] = checkpoint[f"{c_proj_key[:-7]}.bias"]

		if config["scalers"]:	
			conv = [sc_1, sc_2, sc_3, sc_4]
			new[f"transformer.h.{i}.mlp.scalers_fc"]   = torch.tensor(conv)
			new[f"transformer.h.{i}.mlp.scalers_proj"] = torch.tensor(conv)

	return new

device = torch.device("cuda")

# loading GPT2 from HF:
from transformers import GPT2LMHeadModel
gpt 	  = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_sd   = gpt.state_dict()
gpt2_keys = list(gpt2_sd.keys())

print("Loading KronyGPT")
krony_conf = KronyGPTConfig(**config_args)
kronyG = KronyGPT(krony_conf)
krony_sd   = kronyG.state_dict()
k_krony    = krony_sd.keys()


print("Begin decomposition:")
kron_decomp = kron_it_2(gpt2_sd, config_args)

decomp_keys = list(kron_decomp.keys())
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight']

for k in gpt2_keys:
	if any(k.endswith(w) for w in transposed):
		kron_decomp[k] = gpt2_sd[k].t()

rest = [i for i in k_krony if i not in kron_decomp.keys()]

for r_key in rest:
	assert krony_sd[r_key].shape == gpt2_sd[r_key].shape, "dimensions do not match"
	kron_decomp[r_key] = gpt2_sd[r_key]





print("Some dimensions for the eye-bugging:")
i = 5
c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

fc_0   = f"transformer.h.{i}.mlp.c_fc_0"
fc_1   = f"transformer.h.{i}.mlp.c_fc_1"
proj_0 = f"transformer.h.{i}.mlp.c_proj_0"
proj_1 = f"transformer.h.{i}.mlp.c_proj_1"

print(fc_0  , kron_decomp[fc_0].shape)
print(fc_1  , kron_decomp[fc_1].shape)
print(proj_0, kron_decomp[proj_0].shape)
print(proj_1, kron_decomp[proj_1].shape)


print("Some weights")

for lay in range(0):

	gpt_fc_   =  f'transformer.h.{lay}.mlp.c_fc.weight'
	gpt_proj_ =  f'transformer.h.{lay}.mlp.c_proj.weight'

	gpt_fc   =  gpt2_sd[gpt_fc_] 
	gpt_proj =  gpt2_sd[gpt_proj_] 

	fc_0   = kron_decomp[f"transformer.h.{lay}.mlp.c_fc_0"]
	fc_1   = kron_decomp[f"transformer.h.{lay}.mlp.c_fc_1"]

	s_fc = torch.kron(fc_0[0], fc_1[0])
	#for i in range(1,4):
	#	s_fc += torch.kron(fc_0[i], fc_1[i])

	proj_0   = kron_decomp[f"transformer.h.{lay}.mlp.c_proj_0"]
	proj_1   = kron_decomp[f"transformer.h.{lay}.mlp.c_proj_1"]

	s_proj = torch.kron(proj_0[0], proj_1[0])
	#for i in range(1,4):
	#	s_proj += torch.kron(proj_0[i], proj_1[i])

	# Frobenius Norm
	print(f">>>>> For layer {lay}")

	nrm = "fro"
#	nrm = 1

	frobenius_gpt_fc = torch.norm(gpt_fc, nrm).item()
	frobenius_krony_fc = torch.norm(s_fc, nrm).item()

	frobenius_gpt_proj   = torch.norm(gpt_proj, nrm).item()
	frobenius_krony_proj = torch.norm(s_proj, nrm).item()

	#print(">> Before norms:")
	#print(f"Norm: {frobenius_gpt_fc, frobenius_krony_fc}")
	#print(f"Norm: {frobenius_gpt_proj, frobenius_krony_proj}")

	alpha_fc   = frobenius_gpt_fc / frobenius_krony_fc
	alpha_proj = frobenius_gpt_proj / frobenius_krony_proj

	#print("fc_1, and proj_1 before")
	#print(fc_1)
	#print(proj_1)

	#print("fc_1, and proj_1 After: ")
	#print(fc_1)
	#print(proj_1)

	kron_decomp[f"transformer.h.{lay}.mlp.c_fc_1"][0]   = fc_1 * alpha_fc
	kron_decomp[f"transformer.h.{lay}.mlp.c_proj_1"][0] =  proj_1 * alpha_proj


	print("After norms:")
 
	fc_1   = kron_decomp[f"transformer.h.{lay}.mlp.c_fc_1"]
	s_fc = torch.kron(fc_0[0], fc_1[0])
	proj_1   = kron_decomp[f"transformer.h.{lay}.mlp.c_proj_1"]
	s_proj = torch.kron(proj_0[0], proj_1[0])
	
	frobenius_gpt_fc = torch.norm(gpt_fc, nrm).item()
	frobenius_krony_fc = torch.norm(s_fc, nrm).item()

	frobenius_gpt_proj   = torch.norm(gpt_proj, nrm).item()
	frobenius_krony_proj = torch.norm(s_proj, nrm).item()

	print(f"Norm: {frobenius_gpt_fc, frobenius_krony_fc}")
	print(f"Norm: {frobenius_gpt_proj, frobenius_krony_proj}")
 
 
print("3. Saving!")
kronyG.load_state_dict(kron_decomp) 
torch.save(kron_decomp, f"decomps/95M_{dim1}_{dim2}_{factors}_{scalers_}_normal_VL.pt")

"""
	# L1 Norm
	l1_norm = torch.norm(gpt_fc, 1)
	print(f"L1 norm: {l1_norm}")

	# L2 Norm
	l2_norm = torch.norm(A, 2)
	print(f"L2 norm: {l2_norm}")

# test of decomposition of one Kronecker matrix:
def quick_test(k_gpt2, ff: int):
	# Test if the sum of factors of Kroneckers is equal to the original one.
	A = gpt2_sd[k_gpt2]
	ff = 1152
	r1, r2 = kronecker_decompose(A, m=dim1, n=dim2, k=ff)
	s = 0
	for i in range(ff):
		s+= torch.kron(r1[i], r2[i])
	print(r1.shape, r2.shape)
	return s, A
	
if True:
	config0 = dict(
		n_layer=12, 
		n_head=12, 
		n_embd=768,
		vocab_size = 50257,
		block_size = 1024,
		bias = True,
	)
	conf = GPTConfig(**config0)

	import numpy as np

	batch_size = 12
	block_size = 1024

	path = 'data/openwebtext/'
	train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
	val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
	def get_batch(split):
		data = train_data if split == 'train' else val_data
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
		y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
		return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

x, y = get_batch("train")

print("loading gpt normy, karpathy versions")
gpt0 = GPT(conf)
gpt0 = gpt0.from_pretrained("gpt2")
sd0  = gpt0.state_dict()
gpt0.to(device)
kronyG.to(device)

r0 = gpt0(x)
r2 = kronyG(x)

for i in range(12):
    print("Batch > \n")
    print(f"{r0[0][i][0,:10]}")
    print(f"{r2[0][i][0,:10]}")


pass it to kronDecompose python3 kronDecompose.py --origin="ckpt1.pt" --config=config_file
this will automatically generate a checkpoint for you. withe the tag ckpt_p
"""
