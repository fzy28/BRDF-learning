import numpy as np
import torch
from tqdm import tqdm
from utils import *
from model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--niter", default = "10000")
parser.add_argument("--batchsize", default = "2**7")
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

# stratified sampling theta_h, theta_d, phi_d
theta_h_in = np.linspace(0, np.pi / 2, batchsize)
theta_d_in = np.linspace(0, np.pi / 2, batchsize)
phi_d_in = np.linspace(0, np.pi, batchsize)

theta_h_in, theta_d_in, phi_d_in = np.meshgrid(theta_h_in, theta_d_in, phi_d_in)
theta_h_in, theta_d_in, phi_d_in = theta_h_in.flatten(),theta_d_in.flatten(),phi_d_in.flatten()

# compute ground truth
brdf_values = fit_brdf.half_diff_look_up_brdf(theta_h_in, theta_d_in, phi_d_in)
brdf_gt = torch.from_numpy(brdf_values).float().to(device)

# compute input of the network
theta_h_in = torch.from_numpy(theta_h_in).float().to(device)
theta_d_in = torch.from_numpy(theta_d_in).float().to(device)
phi_d_in = torch.from_numpy(phi_d_in).float().to(device)
conca_in = torch.stack([theta_h_in, theta_d_in, phi_d_in], dim=1)

# values_torch = model(conca_in)
# print(conca_in.shape, brdf_values.shape,values_torch.shape)

# loss function, you may play more with it, Lin God!
# I'm not quite sure whether this will influence a lot, I haven't tried it a lot.
criterion = torch.nn.MSELoss()

print("Start training...")
for iteration in (range(niter)):
    # shuffle data
    indices = torch.randperm(conca_in.shape[0]).to(device)
    conca_in = conca_in[indices]
    brdf_gt = brdf_gt[indices]
    
    # forward
    brdf_pred = model(conca_in)
    
    loss = criterion(brdf_pred, brdf_gt)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss
    if iteration % 100 == 0:
        brdf_gt_ = brdf_gt.detach().cpu().numpy()
        brdf_pred_ = brdf_pred.detach().cpu().numpy()
        idx = torch.randint(0, conca_in.shape[0], (1,)).item()
        print("Ground truth: ", brdf_gt_[idx],"idx: ",idx)
        print("Prediction: ", brdf_pred_[idx],"idx: ",idx)
        print("Iteration: %d, Loss: %f" % (iteration, loss.item()))

torch.save(model.state_dict(), "simple_mlp_"+ filename + ".pth")