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

brdf_array = fit_brdf.brdf_array()

r_sum = np.sum(brdf_array[:, :, :,0])
g_sum = np.sum(brdf_array[:, :, :,1])
b_sum = np.sum(brdf_array[:, :, :,2])

r_ratio = r_sum/(r_sum+g_sum+b_sum)
g_ratio = g_sum/(r_sum+g_sum+b_sum)
b_ratio = b_sum/(r_sum+g_sum+b_sum)


all_sum = np.sum(brdf_array,axis=3)

r_array = brdf_array[:,:,:,0]

r_array_ratio = all_sum * r_ratio

n_split = 256

theta_h = np.linspace(0, np.pi / 2, 90)
theta_d = np.linspace(0, np.pi / 2, 90)
phi_d = np.linspace(0, np.pi, 180)


#visualize R channel
fixed_angle = np.pi / 2 * 2

R_pred = r_array_ratio
R_gt = r_array

print(np.mean(np.square(R_pred - R_gt)))
vmin = min(R_pred.min(), R_gt.min())
vmax = max(R_pred.max(), R_gt.max())
levels = np.linspace(vmin, vmax, 100)
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
theta_d, phi_d = np.meshgrid(theta_d, phi_d)

plt.contourf(theta_d, phi_d, R_pred.transpose(), cmap='viridis', levels=levels,vmin=vmin, vmax=vmax)
plt.colorbar(label='pred_fixed_h value')
plt.xlabel('theta_d (x)')
plt.ylabel('phi_d (y)')
plt.title('2D Function Visualization (pred) fixed_angle = ')

plt.subplot(1, 2, 2)
plt.contourf(theta_d, phi_d, R_gt.transpose(), cmap='viridis', levels=levels,vmin=vmin, vmax=vmax)
plt.colorbar(label='gt_fixed_h value')
plt.xlabel('theta_d (x)')
plt.ylabel('phi_d (y)')
plt.title('2D Function Visualization (gt) fixed_angle = ')
plt.show()
