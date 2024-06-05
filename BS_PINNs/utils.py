## Utility functions

import numpy as np
import torch.nn as nn
import copy
from scipy.stats import qmc
from scipy.stats import norm

# from deepxde.backend.set_default_backend import set_default_backend
# set_default_backend("pytorch")

# import deepxde as dde

N = norm.cdf

K = 40
sigma = 0.25
r = 0.05
T = 1
L = 500

# def pde(x, y):
#   K, sigma, r = 4, 0.3, 0.03
#   dy_t = dde.grad.jacobian(y, x, i=0, j=1)
#   dy_x = dde.grad.jacobian(y, x, i=0, j=0)
#   dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#   return dy_t - ((sigma**2 * np.square(x[:,0:1])) / 2) * dy_xx - r * x[:,0:1] * dy_x + r * y

# def payoff_func(x):
#   return np.maximum(K - x[:, 0:1], 0)

# def func(x):
#   temp = K * np.exp(-r * x[:, 1:2])
#   return temp

# def func_r(x):
#   return 0

# def boundary_l(x, on_boundary):
#   return on_boundary and dde.utils.isclose(x[0], 0)

# def boundary_r(x, on_boundary):
#   return dde.utils.isclose(x[0], 10)

# def bs_eq_exact_solution(x, t):
#   """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

#   Parameters
#   ----------
#     x : np.ndarray
#     t : np.ndarray
#   """
#   d1 = (np.log(x/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
#   d2 = d1 - sigma* np.sqrt(t)
#   return K * np.exp(-r*t) * (1 - N(d2)) + (x * (N(d1) - 1))

# def gen_exact_solution():
#   """Generates exact solution for the BS equation for the given values of x and t."""
#   # Number of points in each dimension:
#   x_dim, t_dim = (50, 500)

#   # Bounds of 'x' and 't':
#   x_min, t_min = (0, 0.0)
#   x_max, t_max = (L, 1.0)

#   # Create tensors:
#   t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
#   x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
#   usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

#   # Obtain the value of the exact solution for each generated point:
#   for i in range(x_dim):
#     for j in range(t_dim):
#       usol[i][j] = bs_eq_exact_solution(x[i], t[j])

#   # Save solution:
#   np.savez("bs_eq_data", x=x, t=t, usol=usol)

# def gen_testdata():
#   """Import and preprocess the dataset with the exact solution."""
#   # Load the data:
#   data = np.load("bs_eq_data.npz")
#   # Obtain the values for t, x, and the excat solution:
#   t, x, exact = data["t"], data["x"], data["usol"].T
#   # Process the data and flatten it out (like labels and features):
#   xx, tt = np.meshgrid(x, t)
#   X = np.vstack((np.ravel(xx), np.ravel(tt))).T
#   y = exact.flatten()[:, None]
#   print(f'Shape of X: {X.shape}\nShape of y: {y.shape}\n')
#   return X, y

def get_data(x_range, t_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(t_range[0], t_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)

    b_left = data[0,:,:]
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)
    res = res[res[:,0] != 0]
    res = res[res[:,0] != L]
    res = res[res[:,1] != 0]

    # Computational geometry:
    # geom = dde.geometry.Interval(0, 10)
    # timedomain = dde.geometry.TimeDomain(0, 1)
    # geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # # Initial and boundary conditions:
    # ic = dde.icbc.IC(
    #     geomtime,
    #     payoff_func,
    #     lambda _, on_initial: on_initial,
    # )

    # bc_l = dde.icbc.boundary_conditions.DirichletBC(geom, func, boundary_l)
    # bc_r = dde.icbc.boundary_conditions.DirichletBC(geom, func_r, boundary_r)
    # data = dde.data.TimePDE(
    #     geomtime,
    #     pde,
    #     [bc_r, bc_l, ic],
    #     num_domain=2540,
    #     num_boundary=80,
    #     num_initial=160,
    #     num_test=5,
    #     train_distribution="LHS"
    # )

    # res = data.train_points()[240:]
    # b_left = data.train_points()[:160]  # Initial Points
    # b_right = b_left
    # b_lower = data.train_points()[160:240][data.train_points()[160:240,0] == 0]   # Lower boundary Points
    # b_upper = data.train_points()[160:240][data.train_points()[160:240,0] == 10]  # Upper boundary Points
    return res, b_left, b_right, b_upper, b_lower

def get_test_data(x_range, t_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(t_range[0], t_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)

    b_left = data[0,:,:]
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)
    return res, b_left, b_right, b_upper, b_lower

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower