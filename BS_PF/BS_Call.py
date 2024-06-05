import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import LBFGS, Adam
from tqdm import tqdm
from scipy.stats import norm

from utils import *
from model import PINNsformer

# torch.cuda.empty_cache()

# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# BS params
K = 40
sigma = 0.25
r = 0.05
T = 1
L = 500
N_x = 101
N_t = 101

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = 'cuda:0'   # Change Device Accordingly

# Train PINNsformer
res, b_left, b_right, b_upper, b_lower = get_data([0,L], [0,1], N_x, N_t)
# res_test, _, _, _, _ = get_data([0,10], [0,1], 101, 101)
print(res.shape, b_left.shape, b_right.shape, b_upper.shape, b_lower.shape)

step_size = 1
num_step=10
res = make_time_sequence(res, num_step=num_step, step=step_size)
b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)
print(res.shape, b_left.shape, b_right.shape, b_upper.shape, b_lower.shape)

x_res, t_res = res[:,:,0:1], res[:,:,1:2]
x_left, t_left = b_left[:,:,0:1], b_left[:,:,1:2]
x_right, t_right = b_right[:,:,0:1], b_right[:,:,1:2]
x_upper, t_upper = b_upper[:,:,0:1], b_upper[:,:,1:2]
x_lower, t_lower = b_lower[:,:,0:1], b_lower[:,:,1:2]
# print(x_res.shape, t_res.shape, x_left.shape, t_left.shape)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model = PINNsformer(d_out=1, d_hidden=128, d_model=64, N=1, heads=4).to(device)

model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
# optim = Adam(model.parameters(), lr=1e-6)
print(model)
print(get_n_params(model))

# optim = Adam(model.parameters(), lr=1e-4)

loss_track = []
n_epochs = 101
# print(x_upper[:5])
# print(t_upper[2:4])
for i in tqdm(range(n_epochs)):
  def closure():
    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    # pred_right = model(x_right, t_right)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)

    u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    # print(pred_upper[0])
    loss_res = torch.mean((u_t - ((sigma**2 * x_res**2) / 2) * u_xx - (r * x_res) * u_x + (r * pred_res)) ** 2)
    loss_bc = torch.mean((pred_upper - L) ** 2) + torch.mean((pred_lower) ** 2)
    loss_ic = torch.mean((pred_left[:,0] - torch.max(x_left[:,0] - K, torch.zeros(x_left[:,0].shape).to(device))) ** 2)

    loss_track.append([loss_res.item(), loss_ic.item(), loss_bc.item()])
    loss = loss_res + loss_ic + loss_bc
    optim.zero_grad()
    loss.backward()
    return loss
  
  optim.step(closure)
  if i % 10 == 0:
        print(f'{i}/{n_epochs} PDE Loss: {loss_track[-1][0]:.9f}, BVP Loss: {loss_track[-1][1]:.9f}, IC Loss: {loss_track[-1][2]:.9f},')

print('Loss Res: {:9f}, Loss_BC: {:9f}, Loss_IC: {:9f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))