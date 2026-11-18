# modified from https://github.com/MOR-transport/automated_NsPOD

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import meshgrid
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import gradcheck
from numpy import exp, mod, meshgrid, cos, sin, exp, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky

import os

import sys

impath = "./data/Crossing_waves/"
immpath = "./plots/Crossing_waves/"
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=300, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )

def generate_data(Nx, Nt, coeff1, coeff2, center1, center2, sigma_slope=0.1, sigma_base=4.0, noise=False):
    x = np.arange(0, Nx)
    t = np.linspace(-10,10, Nt)
    t_max = np.max(t)
    [X, T] = np.meshgrid(x, t)
    X = X.T
    T = T.T    

    def gaussian(x, mu, sigma=1.0):
        return np.exp(-0.5*np.power(x - mu, 2.0) / (np.power(sigma, 2.0)))

    q1 = np.zeros_like(X, dtype=np.float64)
    q2 = np.zeros_like(X, dtype=np.float64)
    shift1 = np.polyval(coeff1, t)
    shift2 = np.polyval(coeff2, t)
    for col in range(Nt):
        sigma_t = sigma_base 
        for row in range(Nx):
            q1[row, col] = np.abs(sin(pi * t[col] / t_max)) * gaussian(row, center1 + shift1[col], sigma_t)
            q2[row, col] = np.abs(cos(pi * t[col] / t_max)) * gaussian(row, center2 + shift2[col], sigma_t)
            #q1[row, col] = np.abs(sin(2. * pi * t[col] / t_max)) * gaussian(row, center1 + shift1[col], sigma_t)
            #q2[row, col] = np.abs(cos(2. * pi * t[col] / t_max)) * gaussian(row, center2 + shift2[col], sigma_t)
    
    Q = q1 + q2
    
    if noise:
        Q += np.random.normal(0., 0.1, Q.shape)

    Q /= Q.max()
    
    return Q, x, t

#Nx, Nt, sigma = 401, 200, 40.0
Nx, Nt, sigma = 400, 200, 4.0 # the above matches what we did more closely
coefficients1 = [0.15,0,0.8,1.5]
coefficients2 = [-18,2]
seed = 54 #4 #23 #257 #37 #267 #54
seed = 2 #4 #23 #257 #37 #267 #54
seed = int(sys.argv[1])
print('Seed=%d' % seed)
os.makedirs('plots/positive_burela/seed=%d' % seed, exist_ok=True)
os.makedirs('data/positive_burela/seed=%d' % seed, exist_ok=True)

center_matrix3 = 200

print('Seed:', seed)

np.random.seed(seed)
torch.manual_seed(seed)

Q,x,t = generate_data(Nx, Nt, coefficients1, coefficients2, center_matrix3, center_matrix3, sigma_base=sigma)

X, T = np.meshgrid(x, t)

plt.pcolormesh(X.T, T.T, Q)
plt.colorbar()
plt.xlabel('Position ($x_i$)')
plt.ylabel('Time ($t_j$)')
plt.title('Snapshot Matrix Q')
plt.savefig('snapshot.png', dpi=300)

np.random.seed(seed)
torch.manual_seed(seed)

inputs = np.stack([x.repeat(Nt), np.tile(t, Nx)], axis=1)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
Q_tensor = torch.tensor(Q)

class NuclearNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix):
        ctx.save_for_backward(input_matrix)
        return torch.linalg.matrix_norm(input_matrix, ord="nuc")

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix, = ctx.saved_tensors
        u, s, v = torch.svd(input_matrix, some=False)
        rank = torch.sum(s > 0).item()
        dtype = input_matrix.dtype
        eye_approx = torch.diag((s > 0).to(dtype)[:rank])
        grad_input = torch.matmul(torch.matmul(u[:, :rank], eye_approx), v[:, :rank].t())
        return grad_input * grad_output.unsqueeze(-1).unsqueeze(-1)

class ShapeShiftNet(nn.Module):
    def __init__(self, p_init_coeffs1, p_init_coeffs2):
        super(ShapeShiftNet, self).__init__()

        self.alphas1 = nn.ParameterList(
            [nn.Parameter(torch.tensor([coeff], dtype=torch.float32), requires_grad=True) for coeff in p_init_coeffs1]
        )
        self.alphas2 = nn.ParameterList(
            [nn.Parameter(torch.tensor([coeff], dtype=torch.float32), requires_grad=True) for coeff in p_init_coeffs2]
        )

        self.elu = nn.ELU()

        # Subnetwork for f^1
        self.f1_fc1 = nn.Linear(2, 5)
        self.f1_fc2 = nn.Linear(5, 10)
        self.f1_fc3 = nn.Linear(10, 5)
        self.f1_fc4 = nn.Linear(5, 1)

        # Subnetwork for f^2
        self.f2_fc1 = nn.Linear(2, 5)
        self.f2_fc2 = nn.Linear(5, 10)
        self.f2_fc3 = nn.Linear(10, 5)
        self.f2_fc4 = nn.Linear(5, 1)

    def forward(self, x, t):
        # Pathway for f^1 and shift^1
        # [robert] why polynomial? is a polynomial capable of describing the trajectory?
        shift1 = sum([coeff * t**(3-i) for i, coeff in enumerate(self.alphas1)])

        x_shifted1 = x + shift1
        f1 = self.elu(self.f1_fc1(torch.cat((x_shifted1, t), dim=1)))
        f1 = self.elu(self.f1_fc2(f1))
        f1 = self.elu(self.f1_fc3(f1))
        f1 = self.f1_fc4(f1)

        f1_without_shift = self.elu(self.f1_fc1(torch.cat((x, t), dim=1)))
        f1_without_shift = self.elu(self.f1_fc2(f1_without_shift))
        f1_without_shift = self.elu(self.f1_fc3(f1_without_shift))
        f1_without_shift = self.f1_fc4(f1_without_shift)


        # Pathway for f^2 and shift^2
        # [robert] why is mode 2 a line? this wasn't mentioned in the paper!!
        shift2 = sum([coeff * t**(1-i) for i, coeff in enumerate(self.alphas2)])

        x_shifted2 = x + shift2
        f2 = self.elu(self.f2_fc1(torch.cat((x_shifted2, t), dim=1)))
        f2 = self.elu(self.f2_fc2(f2))
        f2 = self.elu(self.f2_fc3(f2))
        f2 = self.f2_fc4(f2)

        f2_without_shift = self.elu(self.f2_fc1(torch.cat((x, t), dim=1)))
        f2_without_shift = self.elu(self.f2_fc2(f2_without_shift))
        f2_without_shift = self.elu(self.f2_fc3(f2_without_shift))
        f2_without_shift = self.f2_fc4(f2_without_shift)

        return f1, f2, shift1, shift2, f1_without_shift, f2_without_shift

init_coefficients1 = [0.01, 0, 1, 1]
init_coefficients2 = [-1, 1]

model = ShapeShiftNet(init_coefficients1, init_coefficients2)

pretrained_load = False
if pretrained_load:
    state_dict_original = torch.load("Crossing_waves.pth")
    state_dict_new = model.state_dict()

    for name, param in state_dict_original.items():
        if name in state_dict_new:
            state_dict_new[name].copy_(param)
    model.load_state_dict(state_dict_new, strict=False)

def save_data(it):
    x_flat, t_flat = inputs_tensor[:, 0:1], inputs_tensor[:, 1:2]
    optimizer.zero_grad()
    f1_full, f2_full, shift1_pred, shift2_pred, f1_full_nos, f2_full_nos = model(x_flat, t_flat)
    
    combined = f1_full + f2_full
    Q_tilde = combined.view(Nx, Nt).detach().numpy()

    fig, axs = plt.subplots(1, 7, figsize=(24, 6))
    vmin = np.min(Q)
    vmax = np.max(Q)
    
    f1_full = f1_full.detach().numpy()
    f2_full = f2_full.detach().numpy()
    f1_full_nos = f1_full_nos.detach().numpy()
    f2_full_nos = f2_full_nos.detach().numpy()
    shift1_pred = shift1_pred.detach().numpy()
    shift2_pred = shift2_pred.detach().numpy()

    np.savez('data/positive_burela/seed=%d/%d.npz' % (seed, it), f1_full=f1_full, f2_full=f2_full, f1_full_nos=f1_full_nos, f2_full_nos=f2_full_nos, Q_tilde=Q_tilde, shift1_pred=shift1_pred, shift2_pred=shift2_pred)


def plot(it):
    x_flat, t_flat = inputs_tensor[:, 0:1], inputs_tensor[:, 1:2]
    optimizer.zero_grad()
    f1_full, f2_full, shift1_pred, shift2_pred, f1_full_nos, f2_full_nos = model(x_flat, t_flat)
    
    combined = f1_full + f2_full
    Q_tilde = combined.view(Nx, Nt).detach().numpy()

    fig, axs = plt.subplots(1, 7, figsize=(24, 6))
    vmin = np.min(Q)
    vmax = np.max(Q)

    axs[0].pcolormesh(X.T, T.T, Q, vmin=vmin, vmax=vmax)
    axs[0].set_title(r"${\mathbf{Q}}$")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    #Qtilde
    axs[1].pcolormesh(X.T, T.T, Q_tilde, vmin=vmin, vmax=vmax)
    axs[1].set_title(r"$\tilde{\mathbf{Q}}$")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # f^1
    axs[2].pcolormesh(X.T, T.T, f1_full.view(Nx, Nt).detach().numpy(), vmin=vmin)
    axs[2].set_title(r"$\mathcal{T}^1\mathbf{Q}^1$")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("t")
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    # f^2
    axs[3].pcolormesh(X.T, T.T, f2_full.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
    axs[3].set_title(r"$\mathcal{T}^2\mathbf{Q}^2$")
    axs[3].set_xlabel("x")
    axs[3].set_ylabel("t")
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    # f^1
    axs[4].pcolormesh(X.T, T.T, f1_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
    axs[4].set_title(r"$\mathbf{Q}^1$")
    axs[4].set_xlabel("x")
    axs[4].set_ylabel("t")
    axs[4].set_xticks([])
    axs[4].set_yticks([])

    # f^2
    cax4 = axs[5].pcolormesh(X.T, T.T, f2_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
    axs[5].set_title(r"$\mathbf{Q}^2$")
    axs[5].set_xlabel("x")
    axs[5].set_ylabel("t")
    axs[5].set_xticks([])
    axs[5].set_yticks([])
    
    shift_coeffs1 = torch.tensor([p.item() for p in model.alphas1])
    shift_coeffs2 = torch.tensor([p.item() for p in model.alphas2])
    
    shift1 = sum([coeff * t**(3-i) for i, coeff in enumerate(shift_coeffs1)])
    shift2 = sum([coeff * t**(1-i) for i, coeff in enumerate(shift_coeffs2)])
    axs[6].plot(t, shift1, label='coeff_1')
    axs[6].plot(t, shift2, label='coeff_2')
    
    shift1_ex = sum([coeff * t**(3-i) for i, coeff in enumerate(coefficients1)])
    shift2_ex = sum([coeff * t**(1-i) for i, coeff in enumerate(coefficients2)])
    axs[6].plot(t, -shift1_ex, linestyle='--', label='-exact_1')
    axs[6].plot(t, -shift2_ex, linestyle='--', label='-exact_2')
    axs[6].legend()
    axs[6].grid()
    axs[6].set_xlabel('t')
    axs[6].set_ylabel('shift')

    plt.colorbar(cax4, ax=axs.ravel().tolist(), orientation='vertical')
    plt.savefig('plots/positive_burela/seed=%d/%d.png' % (seed, it), dpi=300)

def save_model(it):
    torch.save(model.state_dict(), 'params/seed=%d/%d.pth' % (seed, it))


optimizer = optim.Adam(model.parameters(), lr=0.001)

print('success')

num_epochs = 1000000
lambda_k = 0.5

for epoch in range(1, num_epochs + 1):
    x_flat, t_flat = inputs_tensor[:, 0:1], inputs_tensor[:, 1:2]
    optimizer.zero_grad()
    f1_full, f2_full, shift1_pred, shift2_pred, f1_full_nos, f2_full_nos = model(x_flat, t_flat)
    
    frobenius_loss = torch.norm(Q_tensor - f1_full.view(Nx, Nt) - f2_full.view(Nx, Nt), 'fro') ** 2
    
    nuclear_loss_q1 = NuclearNormAutograd.apply(f1_full_nos.view(Nx, Nt))
    nuclear_loss_q2 = NuclearNormAutograd.apply(f2_full_nos.view(Nx, Nt))
    nuclear_loss = lambda_k*(nuclear_loss_q1 + nuclear_loss_q2)
    
    total_loss = frobenius_loss + nuclear_loss
    
    failed = False
    try:
        total_loss.backward(retain_graph=True)
        optimizer.step()
    except:
        failed = True
    
    shift_coeffs1 = torch.tensor([p.item() for p in model.alphas1])
    shift_coeffs2 = torch.tensor([p.item() for p in model.alphas2])
    
    if failed:
        print('failed :(')
        plot(epoch)
        save_data(epoch)
        break

    if frobenius_loss < 1:
        print("Early stopping is triggered")
        break
    
    if epoch % 100 == 0 or epoch == 1:
        print(
            f'Epoch {epoch}/{num_epochs}, Frob Loss: {frobenius_loss.item()}, Nuclear Loss: {nuclear_loss.item()}, Total loss: {total_loss.item()},'
            f'Coefficients_1:{[coefficients1[i] for i in range(len(coefficients1))]} {shift_coeffs1}, Coefficients_2:{[coefficients2[i] for i in range(len(coefficients2))]} {shift_coeffs2}')
    
    if (epoch % 10000 == 0) or (epoch == 10):
        plot(epoch)
        save_data(epoch)
    #if (epoch % 100000 == 0):
    #    save_model(epoch)
#save_fig(fiilepath=immpath + "Crossing_waves_NN", figure=fig)
