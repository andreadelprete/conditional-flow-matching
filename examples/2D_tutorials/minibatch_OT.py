# The unreasonable positive performance of minibatch optimal transport
import math
import os
import time

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import ot
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.utils import *

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)
total_sample = 1000
# We fix the source and the target distributions for all the examples
x0 = sample_8gaussians(total_sample)
x1 = sample_moons(total_sample)

# We pre-compute the ground cost matrix
M = torch.cdist(x0, x1) ** 2

ot_plan_list = []
num_iter = 20000
a, b = ot.unif(total_sample), ot.unif(total_sample)

print("Compute OTPLAN")
start = time.time()
ot_sampler = OTPlanSampler(method="exact")
pi = ot_sampler.get_map(x0, x1)
ot_plan_list.append(pi)
print("Done computing", time.time()-start)

def mini_batch(data, batch_size):
    """
    Select a subset of sample space according to measure
    with np.random.choice

    Parameters
    ----------
    - data : ndarray(N, d)
    - batch_size : int
        batch size 'm'

    Returns
    -------
    - minibatch : ndarray(ns, nt)
        minibatch of samples
    - sub_weights : ndarray(m,)
        distribution weights of the minibatch
    - id_batch : ndarray(N_data,)
        index of minibatch elements
    """
    id_ = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = ot.unif(batch_size)
    return data[id_], sub_weights, id_

def update_plan(pi, pi_minibatch, id_a, id_b):
    """
    Update the full mini batch transportation matrix

    Parameters
    ----------
    - pi : ndarray(ns, nt)
        full minibatch transportation matrix
    - pi_mb : ndarray(m, m)
        minibatch transportation plan
    - id_a : ndarray(m)
        selected samples from source
    - id_b : ndarray(m)
        selected samples from target

    Returns
    -------
    - pi : ndarray(ns, nt)
        updated transportation matrix
    """
    for i, i2 in enumerate(id_a):
        for j, j2 in enumerate(id_b):
            pi[i2, j2] += pi_minibatch[i][j]
    return pi

def compute_incomplete_plan(xs, xt, a, b, bs, K, C, lambd=1e-1, method="exact"):
    """
    Compute the minibatch gamma with stochastic source and target

    Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - bs : int
        batch size
    - K : int
        number of batch couples
    - C : ndarray(ns, nt)
        cost matrix
    - lambda : float
        entropic reg parameter
    - method : char
        name of method (entropic or emd)

    Returns
    -------
    - incomplete_pi : ndarray(ns, nt)
        incomplete minibatch OT matrix
    """
    incomplete_pi = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    for i in range(K):
        # Select a source and target mini-batch couple
        sub_xs, sub_weights_a, id_a = mini_batch(xs, bs)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, bs)

        # compute ground cost between minibatches
        mb_C = C[id_a, :][:, id_b]
        # The minibatch Cost could be computed on the fly instead of using the full-size ground cost

        # Solve the OT problem between minibatches
        if method == "exact":
            G0 = ot.emd(sub_weights_a, sub_weights_b, mb_C.copy())

        elif method == "entropic":
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, mb_C, lambd)

        # Update the transport plan
        incomplete_pi = update_plan(incomplete_pi, G0, id_a, id_b)

    return (1 / K) * incomplete_pi

# Compute incomplete mbot plan for several batch sizes
batch_size_list = [128, 64]
for i, batch_size in enumerate(batch_size_list):
    print("Compute incomplete mbot plan for batch size", batch_size)
    start = time.time()
    incomplete_mbot_plan = compute_incomplete_plan(x0, x1, a, b, batch_size, num_iter, M.numpy())
    ot_plan_list.append(incomplete_mbot_plan)
    print("Done computing", time.time()-start)

pl.figure(figsize=(12, 5))
for i in range(3):
    pl.subplot(1, 3, i + 1)
    pl.imshow(ot_plan_list[i], interpolation="nearest")
    pl.axis("off")
    if i == 0:
        pl.title("True OT plan")
    else:
        batch_size = batch_size_list[i - 1]
        pl.title("Incomplete MBOT plan (batch size:{})".format(batch_size), fontsize=14)
    pl.tight_layout()

sigma = 0.1
dim = 2
pl.figure(3, figsize=(12, 10))
batch_size = 128
for i, mbot_plan in enumerate(ot_plan_list):
    print("Start training with MBOT PLAN n.", i)
    model = MLP(dim=dim, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)

    ot_sampler = OTPlanSampler(method="exact")

    start = time.time()
    for k in range(20000):
        optimizer.zero_grad()

        indices_i, indices_j = ot_sampler.sample_map(
            mbot_plan, batch_size=batch_size, replace=False
        )
        batch_x0, batch_x1 = x0[indices_i], x1[indices_j]

        t, xt, ut = FM.sample_location_and_conditional_flow(batch_x0, batch_x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

    end = time.time()
    print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
    start = end
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            x0,
            t_span=torch.linspace(0, 1, 100),
        )

        pl.subplot(2, 3, i + 1)
        pl.imshow(ot_plan_list[i], interpolation="nearest")
        pl.axis("off")
        if i == 0:
            pl.title("True OT plan", fontsize=14)
        else:
            batch_size = batch_size_list[i - 1]
            pl.title("Incomplete MBOT plan (batch size:{})".format(batch_size), fontsize=14)

        plt.subplot(2, 3, i + 4)
        traj = traj.cpu().numpy()
        plt.scatter(
            traj[0, :total_sample, 0], traj[0, :total_sample, 1], s=10, alpha=0.8, c="black"
        )
        plt.scatter(
            traj[:, :total_sample, 0], traj[:, :total_sample, 1], s=0.2, alpha=0.2, c="olive"
        )
        plt.scatter(traj[-1, :total_sample, 0], traj[-1, :total_sample, 1], s=4, alpha=1, c="blue")
        plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
        plt.xticks([])
        plt.yticks([])
        pl.axis("off")
        if i == 0:
            pl.title("OT-CFM trained with true OT plan", fontsize=15)
        else:
            pl.title("OT-CFM trained w. MBOT (size:{})".format(batch_size), fontsize=15)
        pl.tight_layout()
# pl.savefig('OT_CFM_different_MBOT_plan.png')

reg_list = [1, 10, 100]

pl.figure(3, figsize=(12, 5))

x0 = sample_8gaussians(1000)
x1 = sample_moons(1000)

ot_plan_list = []

for i, reg in enumerate(reg_list):
    ot_sampler = OTPlanSampler(method="sinkhorn", reg=reg)

    entropic_ot_plan = ot_sampler.get_map(x0, x1)
    ot_plan_list.append(entropic_ot_plan)
    pl.subplot(1, 3, i + 1)
    pl.imshow(entropic_ot_plan, interpolation="nearest")
    pl.axis("off")
    pl.title("Entropic OT plan (reg:{})".format(reg), fontsize=14)
    pl.tight_layout()

total_sample = 1000
sigma = 0.1
dim = 2
batch_size = 128

pl.figure(3, figsize=(12, 10))

for i, reg in enumerate(reg_list):
    model = MLP(dim=dim, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)

    ot_sampler = OTPlanSampler(method="sinkhorn", reg=reg)

    pi = ot_sampler.get_map(x0, x1)

    start = time.time()
    for k in range(20000):
        optimizer.zero_grad()

        indices_i, indices_j = ot_sampler.sample_map(pi, batch_size=batch_size, replace=False)
        batch_x0, batch_x1 = x0[indices_i], x1[indices_j]

        t, xt, ut = FM.sample_location_and_conditional_flow(batch_x0, batch_x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

    end = time.time()
    print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
    start = end
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            x0,
            t_span=torch.linspace(0, 1, 100),
        )
        pl.subplot(2, 3, i + 1)
        pl.imshow(ot_plan_list[i], interpolation="nearest")
        pl.axis("off")
        pl.title("Entropic-OT plan (reg={})".format(reg), fontsize=18)

        plt.subplot(2, 3, i + 4)
        traj = traj.cpu().numpy()
        plt.scatter(
            traj[0, :total_sample, 0], traj[0, :total_sample, 1], s=10, alpha=0.8, c="black"
        )
        plt.scatter(
            traj[:, :total_sample, 0], traj[:, :total_sample, 1], s=0.2, alpha=0.2, c="olive"
        )
        plt.scatter(traj[-1, :total_sample, 0], traj[-1, :total_sample, 1], s=4, alpha=1, c="blue")
        plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
        plt.xticks([])
        plt.yticks([])
        pl.axis("off")
        pl.title("OT-CFM trained w. E-OT plan (reg={})".format(reg), fontsize=14)
        pl.tight_layout()
pl.savefig("OT_CFM_different_entropic_OT_plan.png")
