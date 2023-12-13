import numpy as np
import torch
from tqdm import tqdm
from utils import *
from model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--niter", default = "400000")
parser.add_argument("--batchsize", default = "2**21")
parser.add_argument("--filename", default = "alum-bronze.binary")
args = parser.parse_args()

niter = eval(args.niter)
batchsize = eval(args.batchsize)
filename = args.filename

# load measured brdf
fit_brdf = MeasuredBRDF(filename)

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model_path = "be_simple_mlp_"+ filename + ".pth"
# model.load_state_dict(torch.load(model_path))

n_split = 360
# stratified sampling theta_h, theta_d, phi_d
theta_h_in = np.linspace(0, np.pi / 2, n_split )
# theta_d_in_coarse = np.linspace(0, np.pi / 2 * 1.2 / 1.6, 90)
# theta_d_in_fine = np.linspace(np.pi / 2 * 1.2 / 1.6, np.pi / 2, 90)
# theta_d_in = np.concatenate([theta_d_in_coarse, theta_d_in_fine])
theta_d_in = np.linspace(0, np.pi / 2, n_split)
phi_d_in = np.linspace(0, np.pi, n_split * 2)

theta_h_in, theta_d_in, phi_d_in = np.meshgrid(theta_h_in, theta_d_in, phi_d_in)
theta_h, theta_d, phi_d = theta_h_in.flatten(),theta_d_in.flatten(),phi_d_in.flatten()


we_eval = True
N_data = theta_h.shape[0]
print("Start training...")
for iteration in (range(niter)):
    indices = np.random.randint(0, N_data, batchsize)
    theta_h_in = theta_h[indices] + np.random.rand(batchsize) * np.pi / n_split / 2
    theta_d_in = theta_d[indices] + np.random.rand(batchsize) * np.pi / n_split / 2
    phi_d_in = phi_d[indices] + np.random.rand(batchsize) * np.pi / n_split
    
    # compute ground truth
    brdf_values = fit_brdf.half_diff_look_up_brdf(theta_h_in, theta_d_in, phi_d_in)
    brdf_gt = torch.from_numpy(brdf_values).float().to(device)

    # compute input of the network
    theta_h_in = torch.from_numpy(theta_h_in).float().to(device)
    theta_d_in = torch.from_numpy(theta_d_in).float().to(device)
    phi_d_in = torch.from_numpy(phi_d_in).float().to(device)
    conca_in = torch.stack([theta_h_in, theta_d_in, phi_d_in], dim=1)
    
    # forward
    brdf_pred = model(conca_in)
    
    loss = torch.mean(torch.abs(brdf_pred - brdf_gt))

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss
    if iteration % 100 == 0 and we_eval:
        brdf_gt_ = brdf_gt.detach().cpu().numpy()
        brdf_pred_ = brdf_pred.detach().cpu().numpy()
        idx = 90000
        print("Ground truth: ", brdf_gt_[idx],"idx: ",idx)
        print("Prediction: ", brdf_pred_[idx],"idx: ",idx)
        print("Iteration: %d, Loss: %f" % (iteration, loss.item()))
    if iteration % 10000 == 0:
        torch.save(model.state_dict(), "be_simple_mlp_"+ filename + ".pth")