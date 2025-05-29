import math
import os
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from cleanfid import fid

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

def exact_div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = torch.func.jacrev(u)
    return lambda x, *args: torch.trace(J(x))

class cnf_wrapper(torch.nn.Module):
    """Wraps model to a torchdyn compatible CNF format.
    Appends an additional dimension representing the change in likelihood
    over time.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.div_fn, self.eps_fn = self.get_div_and_eps()

    def get_div_and_eps(self):
        return exact_div_fn, None

    def forward(self, t, x, *args, **kwargs):
        t = t.squeeze()
        x = x[..., :-1]

        def vecfield(y):
            return self.model(torch.cat([y, t[None]]))

        if self.eps_fn is None:
            div = torch.vmap(self.div_fn(vecfield))(x)
        else:
            div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        dx = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        return torch.cat([dx, div[:, None]], dim=-1)
    

def plot_my_trajectories(p0, model, x1, plot_size):
    """Plot trajectories of some selected samples."""
    node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", 
                atol=1e-4, rtol=1e-4)
    traj = node.trajectory(p0.sample([plot_size]),
                        t_span=torch.linspace(0, 1, 100),).cpu().numpy()
    # evaluate FID score
    # x1_gen = traj[-1,:,:]
    # fid_score = fid.fid_from_feats(x1_gen, x1_plot)
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.6, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.1, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=0.6, c="blue")
    plt.scatter(x1[:,0], x1[:,1], s=4, alpha=0.6, c="red")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)", "Targets"])
    plt.grid(True)

def compute_neg_log_likelihood(x1, p0, model):
    # evaluate negative log likelihood that sampling p0 and integrating the given vector field
    # (model) we get the given samples x1.
    cnf = NeuralODE(
        cnf_wrapper(model), solver="dopri5", sensitivity="adjoint", 
        atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        x1_with_ll = torch.cat([x1, torch.zeros(x1.shape[0], 1)], dim=-1)
        x0_with_ll = cnf.trajectory(x1_with_ll, t_span=torch.linspace(1, 0, 100))[-1]
        logprob = p0.log_prob(x0_with_ll[..., :-1]) + x0_with_ll[..., -1]
        nll_score = -torch.mean(logprob)
    return nll_score
    
def sample_target_distribution(size):
    if(DATASET=="moons"):
        return sample_moons(size)
    return p1.sample([size])

def get_test_name(method, buffer_size, vf_batch_size):
    if(method=="NOVEL"):
        return method+"_buff_"+str(buffer_size)+"_vfbs_"+str(vf_batch_size)
    return method

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)
SAVE_DATA = 0
LOAD_DATA = 0
METHOD = "NOVEL" # STANDARD_CFM or OPTIMAL_TRANSPORT or NOVEL
DATASET = "moons"
batch_size = 128
TRAINING_EPOCHS = 30000

# size of the minibatch used for estimating the marginal vector field
VF_BATCH_SIZES = [64] # 32 or 64 both worked very well, at least 16 to get decent results
# VF_BATCH_SIZES = [16, 32, 64] # at least 16

# size of the training buffer
BUFFER_SIZES = [int(10000/64)]
# BUFFER_SIZES = [int(10000/32),
#                 int(10000/64),
#                 int(10000/128)]

NTESTS = int(TRAINING_EPOCHS/30)
DO_PLOTS = False
sigma = 0.01
dim = 2
test_size = 1024

ot_sampler = OTPlanSampler(method="exact")
p0 = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
p1 = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 3])),
    torch.distributions.Independent(torch.distributions.Normal(
        torch.tensor([[-5., -4], [6., 9.]]), torch.tensor([[1., 3.], [2., 1.]])), 1)
)
CFM = LowVarianceConditionalFlowMatcher(p0, sigma)

if(LOAD_DATA):
    nll_scores = pickle.load(open("nll_scores_novel", "rb"))
    epochs = pickle.load(open("epochs_novel", "rb"))
else:
    epochs, nll_scores = {}, {}
    t_vs_p = []
    x1_test = sample_target_distribution(test_size)
    for vf_batch_size in VF_BATCH_SIZES:
        data_buffer = []
        buffer_size = np.max(BUFFER_SIZES)
        # First fill out the data buffer 
        print(f"Prepare data buffer of {buffer_size} elements")
        print("Using training method", METHOD)
        start = time.time()
        if(METHOD == "OPTIMAL_TRANSPORT" or METHOD=="STANDARD_CFM"):
            for k in range(TRAINING_EPOCHS):
                x0 = p0.sample([batch_size])
                x1 = sample_target_distribution(batch_size)
                # Draw samples from OT plan
                if(METHOD == "OPTIMAL_TRANSPORT"):
                    x0, x1 = ot_sampler.sample_plan(x0, x1)
                t, xt, ut = CFM.sample_location_and_conditional_flow(x0, x1)
                data_buffer.append([t, xt, ut])
        else:
            print(f"Using VF batch size {vf_batch_size}")
            for k in range(int(buffer_size)):
                x0 = p0.sample([batch_size])
                x1 = sample_target_distribution(batch_size)
                x1_vf_batch = sample_target_distribution(vf_batch_size)
                t, xt, ut = CFM.sample_location_and_conditional_flow(x0, x1, x1_vf_batch)
                data_buffer.append([t, xt, ut])
        data_prep_time = time.time()-start
        print(f"Done preparing data buffer. It took {data_prep_time:0.3f} s")

        for buffer_size in BUFFER_SIZES:
            # use only the first buffer_size elements of the buffer
            test_name = get_test_name(METHOD, buffer_size, vf_batch_size)
            epochs[test_name] = []
            nll_scores[test_name] = []

            # Then train the network using the data in the buffer
            print("Start training with buffer size", buffer_size, "and VF batch size", vf_batch_size)
            model = MLP(dim=dim, time_varying=True)
            optimizer = torch.optim.Adam(model.parameters())
            start = time.time()
            for k in range(TRAINING_EPOCHS):
                [t, xt, ut] = data_buffer[np.random.randint(buffer_size)]
                vt = model(torch.cat([xt, t[:, None]], dim=-1))
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (k + 1) % NTESTS == 0:
                    end = time.time()
                    with torch.no_grad():
                        # evaluate negative log likelihood
                        nll_score = compute_neg_log_likelihood(x1_test, p0, model)
                        epochs[test_name].append(k)
                        nll_scores[test_name].append(nll_score)
                        print(f"{k+1}: loss {loss.item():0.3f} NLL {nll_score:0.3f} time {(end - start):0.2f}")
                        if(DO_PLOTS):
                            plot_my_trajectories(p0, model, x1_test, test_size)
                    start = time.time()
    if(SAVE_DATA):
        pickle.dump(nll_scores, open("nll_scores_novel", "wb"))
        pickle.dump(epochs, open("epochs_novel", "wb"))

def filter_data(v):
    out = len(v)*[0.0,]
    for i in range(1, len(v)-1):
        out[i] = (v[i-1] + 2*v[i] + v[i+1])*0.25
    out[0] = (3*v[0] + v[1])*0.25
    out[-1] = (v[-2] + 3*v[-1])*0.25
    return out

if(METHOD=="NOVEL"):
    for vf_batch_size in VF_BATCH_SIZES:
        plt.figure()
        for k in epochs.keys():
            if(str(vf_batch_size) in k):
                plt.plot(epochs[k], filter_data(nll_scores[k]), label=k)
        data = np.load("nll_standard_cfm.npz")
        plt.plot(data["epochs"], filter_data(data["nnl_scores"]), label="Classic")
        plt.legend()
        plt.ylabel("Negative Log Likelihood")
        plt.xlabel("Training epochs")
        plt.grid(True)

    for buffer_size in BUFFER_SIZES:
        plt.figure()
        for k in epochs.keys():
            if(str(buffer_size) in k):
                plt.plot(epochs[k], filter_data(nll_scores[k]), label=k)
        data = np.load("nll_standard_cfm.npz")
        plt.plot(data["epochs"], filter_data(data["nnl_scores"]), label="Classic")
        plt.legend()
        plt.ylabel("Negative Log Likelihood")
        plt.xlabel("Training epochs")
        plt.grid(True)
else:
    plt.figure()
    for k in epochs.keys():
        plt.plot(epochs[k], filter_data(nll_scores[k]), label=k)
    data = np.load("nll_standard_cfm.npz")
    plt.plot(data["epochs"], filter_data(data["nnl_scores"]), label="Classic")
    plt.legend()
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Training epochs")
    plt.grid(True)

# if(USE_STANDARD_CFM):
#     np.savez("nll_standard_cfm.npz", epochs=epochs, nnl_scores=nll_scores)
# else:
#     np.savez("nll_novel_cfm.npz", epochs=epochs, nnl_scores=nll_scores)
