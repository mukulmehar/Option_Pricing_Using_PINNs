import sys

from zmq import device
print(sys.executable)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
from scipy.stats import norm

from utils import *
from pinn import PINNs

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

device = 'cuda:1'

res, b_left, b_right, b_upper, b_lower = get_data([0, L], [0, 1], N_x, N_t)
# res_test, _, _, _, _ = get_data([0,10], [0,1], N_x, N_t)
# print(res.shape, b_left.shape, b_right.shape, b_upper.shape, b_lower.shape)
res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)
x_res, t_res = res[:,0:1], res[:,1:2]
x_left, t_left = b_left[:,0:1], b_left[:,1:2]
x_right, t_right = b_right[:,0:1], b_right[:,1:2]
x_upper, t_upper = b_upper[:,0:1], b_upper[:,1:2]
x_lower, t_lower = b_lower[:,0:1], b_lower[:,1:2]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model = PINNs(in_dim=2, hidden_dim=256, out_dim=1, num_layer=6).to(device)

model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
# optim = Adam(model.parameters(), lr=1e-4)

n_params = get_n_params(model)

print(model)
print(n_params)

loss_track = []
n_epochs = 101

for i in tqdm(range(n_epochs)):
  def closure():
    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    pred_right = model(x_right, t_right)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)

    u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

    # print(torch.mean((u_t - ((sigma**2 * x_res**2) / 2) * u_xx - (r * x_res) * u_x + (r * pred_res)) ** 2))
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

state = {
    'epoch': n_epochs,
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict(),
    'loss_hist': loss_track
}

# torch.save(state, './BS_Put_PINNs_101')

# Testing
N_x=201
N_t=201
res_test, _, b_right_test, _, _ = get_test_data([0,10], [0,1], N_x, N_t)
# step_size = 1e-4

N = norm.cdf

# res_test = make_time_sequence(res_test, num_step=5, step=step_size)
res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
b_right_test = torch.tensor(b_right_test, dtype=torch.float32, requires_grad=True).to(device)
x_test, t_test = res_test[:,0:1], res_test[:,1:2]
x_right_test, t_right_test = b_right_test[:,0:1], b_right_test[:,1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:,0:1]
    pred_right = model(x_right_test, t_right_test)[:,0:1]

    pred = pred.cpu().detach().numpy()
    pred_right = pred_right.cpu().detach().numpy()

pred = pred.reshape(N_x,N_t)
# print(pred_right.shape)

def BS_CALL(S, T):
    d1 = (torch.log(S/K) + (r + sigma**2 / 2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    return S * N(d1) - K * torch.exp(-r*T)* N(d2)

def BS_PUT(S, T):
    d1 = (np.log(np.where(S/K > 1e-8, S/K, 1e-8)) + (r + sigma**2/2)*T) / (sigma*np.sqrt(np.where(T > 1e-8, T, 1e-8)))
    d2 = d1 - sigma* np.sqrt(T)
    return K * np.exp(-r*T) * (1 - N(d2)) + S * (N(d1) - 1)

res_test, _, b_right_test, _, _ = get_test_data([0,10], [0,1], N_x, N_t)
u = BS_PUT(res_test[:,0], res_test[:,1]).reshape(N_x,N_t)
u_right = BS_PUT(b_right_test[:,0], b_right_test[:,1])

# Relative l1 and l2 errors at full grid
rl1 = np.sum(np.abs(u-pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u-pred)**2) / np.sum(u**2))
print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

rl1_right = np.sum(np.abs(u_right-pred_right[:,0])) / np.sum(np.abs(u_right))
rl2_right = np.sqrt(np.sum((u_right-pred_right[:,0])**2) / np.sum(u_right**2))
print('relative L1 error (At Final Time) :{:4f}'.format(rl1_right))
print('relative L2 error (At Final Time) :{:4f}'.format(rl2_right))

plt.figure(figsize=(4,3))
plt.imshow(pred, extent=[0,10,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
# plt.savefig('./1dBS_Put_pinns_pred.png')

plt.figure(figsize=(4,3))
plt.imshow(u, extent=[0,10,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
# plt.savefig('./1dBS_Put_exact.png')

# plt.plot(X[final_index, 0], y_pred[final_index], '--', color="r")
plt.figure()
plt.plot(x_right_test.cpu().detach().numpy(), pred_right, '--', color="r")
plt.xlabel('S')
plt.ylabel('V(S, T)')
plt.title('Predicted u(x,t) (Final Time)')
# set the limits
plt.xlim([0, 10])
plt.ylim([0, 4])
# plt.savefig('./1dBS_Put_pinns_pred(Final Time).png')

# Pointwise Error at final time
plt.figure()
plt.plot(x_right_test.cpu().detach().numpy()[:,0], u_right - pred_right[:,0], '--', color="r")
plt.xlabel('S')
plt.ylabel('V(S, T)')
plt.title('Pointwise Error (Final Time)')
# set the limits
plt.xlim([0, 10])
plt.ylim([0, 0.002])
# plt.savefig('./1dBS_Put_PINNs_pointwise_error(Final Time).png')

print("Maximum Poinwise error (At Final Fime): {:4f}".format(np.max(np.abs(u_right - pred_right[:,0]))))