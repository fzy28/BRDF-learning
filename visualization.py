## test the learned model and visualize the results
import numpy as np
import torch
from tqdm import tqdm
from utils import *
from model import *
import matplotlib.pyplot as plt

filename = "alum-bronze.binary"
model_path = "simple_mlp_"+ filename + ".pth"

fit_brdf = MeasuredBRDF(filename)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

n_split = 256

theta_h = np.linspace(0, np.pi / 2, n_split)
theta_d = np.linspace(0, np.pi / 2, n_split)
phi_d = np.linspace(0, np.pi, n_split)

plane1_phi_d, plane2_phi_d = np.meshgrid(theta_h, theta_d)
plane1_theta_d, plane2_theta_d = np.meshgrid(theta_h, phi_d)
plane1_theta_h, plane2_theta_h = np.meshgrid(theta_d, phi_d)

plane1_phi_d, plane2_phi_d = plane1_phi_d.flatten(), plane2_phi_d.flatten()
plane1_theta_d, plane2_theta_d = plane1_theta_d.flatten(), plane2_theta_d.flatten()
plane1_theta_h, plane2_theta_h = plane1_theta_h.flatten(), plane2_theta_h.flatten()


fixed_angle = "np.pi * 0.3"
repeated_array = np.full(n_split**2, eval(fixed_angle))



theta_h_in = torch.from_numpy(repeated_array).float().to(device)
theta_d_in = torch.from_numpy(plane1_theta_h).float().to(device)
phi_d_in = torch.from_numpy(plane2_theta_h).float().to(device)
conca_in_fixed_h = torch.stack([theta_h_in, theta_d_in, phi_d_in], dim=1)

gt_fixed_h = fit_brdf.half_diff_look_up_brdf(
    repeated_array, plane1_theta_h, plane2_theta_h, use_interpolation=False
)
pred_fixed_h = model(conca_in_fixed_h).detach().cpu().numpy()

gt_fixed_h = gt_fixed_h.reshape(n_split, n_split,3)
pred_fixed_h = pred_fixed_h.reshape(n_split, n_split,3)


#visualize R channel

R_pred = pred_fixed_h[:,:,0]
R_gt = gt_fixed_h[:,:,0]
vmin = min(R_pred.min(), R_gt.min())
vmax = max(R_pred.max(), R_gt.max())
levels = np.linspace(vmin, vmax, 100)
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.contourf(theta_d, phi_d, R_pred, cmap='viridis', levels=levels,vmin=vmin, vmax=vmax)
plt.colorbar(label='pred_fixed_h value')
plt.xlabel('theta_d (x)')
plt.ylabel('phi_d (y)')
plt.title('2D Function Visualization (pred) fixed_angle = '+ fixed_angle)

plt.subplot(1, 2, 2)
plt.contourf(theta_d, phi_d, R_gt, cmap='viridis', levels=levels,vmin=vmin, vmax=vmax)
plt.colorbar(label='gt_fixed_h value')
plt.xlabel('theta_d (x)')
plt.ylabel('phi_d (y)')
plt.title('2D Function Visualization (gt) fixed_angle = '+fixed_angle)
plt.show()

