import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

from imagenet_data import MyDataset
from torch.utils.data import DataLoader

'''
Integrate the marginal vector field (VF) without training it, but  estimating it
with our novel method. Check if the VF becomes straight after a while.
'''

def sample_target_distribution(size):
    if(DATASET=="moons"):
        return sample_moons(size)
    elif(DATASET=="cifar"):
        if(size==vf_batch_size):
            x1, _ = next(iter(dataloader_vf))
        else:
            raise Exception("Specified size is not available!")
        x1 = x1 * 2.0 - 1.0
        x1 = x1.to("cpu").view(size, -1)
        return x1
    return p1.sample([size])

def compute_marginal_vf(x, t, x1_batch, p0, sigma):
    deno = 1-(1-sigma)*t
    # computation of the x0's corresponding to xt and the dataset of x1
    x0_xbar = (x - t*x1_batch)/deno
    # computation of the prob associated to these values of x0
    logprob = p0.log_prob(x0_xbar)
    max_logprob, _ = torch.max(logprob, dim=0, keepdim=True)
    logprob = logprob - max_logprob
    prob = torch.exp(logprob)
    norm_prob = prob / torch.sum(prob).item()
    # estimate the vector field
    v_batch = (x1_batch - (1-sigma)*x) / deno
    u = torch.sum(norm_prob * v_batch.T, dim=1)
    return u

DATASET = "moons"
DATASET = "cifar"

N_TESTS = 10
T_LOOKAHEAD = 0.4
# size of the minibatch used for estimating the marginal vector field
vf_batch_size = 4*1024 # 32 or 64 both worked very well, at least 16 to get decent results
# size of the training buffer
sigma = 0.01


if(DATASET=="cifar"):
    dataset = MyDataset(dataset="cifar10", data_path="data")
    # set VF batch size equal to number of data points
    vf_batch_size = dataset.dataset.data.shape[0]
    
    dataloader_vf = DataLoader(dataset, batch_size=vf_batch_size, shuffle=True)
    dim = 3072
else:
    dim = 2
    p1 = torch.distributions.mixture_same_family.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([1, 3])),
        torch.distributions.Independent(torch.distributions.Normal(
            torch.tensor([[-5., -4], [6., 9.]]), torch.tensor([[1., 3.], [2., 1.]])), 1)
    )

p0 = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
# CFM = LowVarianceConditionalFlowMatcher(p0, sigma)
CFM = ConditionalFlowMatcher(sigma)

# First fill out the data buffer 
start = time.time()
print(f"Using VF batch size {vf_batch_size}")

# integration step
dt = 0.05
N = int(1/dt)
u_diff = []
for k in range(N_TESTS):
    print("Test n.", k)
    t = 0.0 # initialize time
    x0 = p0.sample([1]) # sample initial state
    # sample target dataset
    x1_batch = sample_target_distribution(vf_batch_size)
    
    # initialize lists to store data
    xt = [x0]
    ut = []

    # start integration loop
    for i in range(N):
        # print("Integration time", t)
        u = compute_marginal_vf(xt[i], t, x1_batch, p0, sigma)
        x_new = xt[i] + dt*u
        t += dt
        xt.append(x_new)
        ut.append(u)
        if(i>0):
            delta_u = (ut[i]-ut[i-1]).norm()
            u_diff.append([t, delta_u])
    
    # compute integration lookahead at t=0.3
    N_LOOKAHEAD = int(T_LOOKAHEAD/dt)
    # print("N_LOOKAHEAD", N_LOOKAHEAD)
    x_lookahead = xt[N_LOOKAHEAD] + (1-T_LOOKAHEAD)*ut[N_LOOKAHEAD]
    lookahead_err = (xt[-1] - x_lookahead).norm()
    print("Lookahead error", lookahead_err)

data_prep_time = time.time()-start
print(f"Done computing. It took {data_prep_time:0.3f} s")

u_diff = np.array(u_diff)
plt.figure()
plt.plot(u_diff[:,0], u_diff[:,1], ' x', alpha=0.8)
plt.xlabel("Time")
plt.ylabel("Delta VF")
plt.yscale("log")
plt.grid(True)
plt.show()

synthetic_samples = torch.clamp(
                    xt[-1] * 0.5 + 0.5, min=0.0, max=1.0
                )
synthetic_samples = torch.floor(synthetic_samples * 255)
import torchvision
synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
grid = torchvision.utils.make_grid(
    synthetic_samples[:16], nrow=4, normalize=False
)