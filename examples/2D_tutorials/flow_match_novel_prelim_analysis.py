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

def sample_target_distribution(size):
    if(DATASET=="moons"):
        return sample_moons(size)
    elif(DATASET=="cifar"):
        if(size==batch_size):
            x1, _ = next(iter(dataloader))
        elif(size==vf_batch_size):
            x1, _ = next(iter(dataloader_vf))
        else:
            raise Exception("Specified size is not available!")
        x1 = x1 * 2.0 - 1.0
        x1 = x1.to("cpu").view(size, -1)
        return x1
    return p1.sample([size])

DATASET = "moons"
DATASET = "cifar"

SCORE = "normalized_weight"
SCORE = "VF_diff"

batch_size = 128
# size of the minibatch used for estimating the marginal vector field
vf_batch_size = 2048 # 32 or 64 both worked very well, at least 16 to get decent results
# size of the training buffer
buffer_size = 10
sigma = 0.01

if(DATASET=="cifar"):
    dataset = MyDataset(dataset="cifar10", data_path="data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
print(f"Prepare buffer of size {buffer_size}")
start = time.time()
print(f"Using VF batch size {vf_batch_size}")

t_vs_score = []
for k in range(buffer_size):
    print("Iter", k)
    x0 = p0.sample([batch_size])
    x1 = sample_target_distribution(batch_size)

    t, xt, ut = CFM.sample_location_and_conditional_flow(x0, x1)
    deno = 1-(1-sigma)*t
    x1_batch = sample_target_distribution(vf_batch_size)
    for i in range(batch_size):            
        # make sure that the value of x1 that generated xt[i] is in the batch to avoid 
        # having all probabilities equal to zero
        x1_batch[0] = x1[i]
        # batch computation of the values of x0 corresponding to xt[i] and the dataset of x1
        x0_xbar = (xt[i] - t[i]*x1_batch)/deno[i]
        # batch computation of the probabilities associated to these values of x0
        logprob = p0.log_prob(x0_xbar)
        max_logprob, _ = torch.max(logprob, dim=0, keepdim=True)
        logprob = logprob - max_logprob
        prob = torch.exp(logprob)
        norm_prob = prob / torch.sum(prob).item()
        # estimate the vector field
        v_batch = (x1_batch - (1-sigma)*xt[i]) / deno[i]
        ut[i] = torch.sum(norm_prob * v_batch.T, dim=1)
        if(SCORE=="normalized_weight"):
            norm_score = 1-norm_prob[0].item()
        else:
            u_cond = v_batch[0]
            norm_score = (ut[i] - u_cond).norm() / u_cond.norm()
        t_vs_score.append([t[i], norm_score])
data_prep_time = time.time()-start
print(f"Done computing. It took {data_prep_time:0.3f} s")

t_vs_score = np.array(t_vs_score)
plt.figure()
plt.plot(t_vs_score[:,0], t_vs_score[:,1], ' x', alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Normalized score")
plt.yscale("log")
plt.grid(True)
plt.show()