import numpy as np
import torch
from tqdm import tqdm
from utils import *
from model import *
import argparse
from nps.nps import NPsDecoder
parser = argparse.ArgumentParser()

parser.add_argument("--niter", default = "400000")
parser.add_argument("--batchsize", default = "2**18")
parser.add_argument("--filename", default = "alum-bronze_latentVector")
args = parser.parse_args()

niter = eval(args.niter)
batchsize = eval(args.batchsize)
filename = args.filename

# load measured brdf

NPs_model = NPsDecoder()
NPs_model.load_state_dict(torch.load('nps/model/decoder.pt'))

NPs_model.to('cuda').eval()

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Nps().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

z = np.load("./nps/latent_vectors/" + filename + ".npy")
z = torch.from_numpy(z).to(device).float()
z = z.repeat(1,batchsize, 1)
n_split = 180
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
    # compute input of the network
    theta_h_in = torch.from_numpy(theta_h_in).float().to(device)
    theta_d_in = torch.from_numpy(theta_d_in).float().to(device)
    phi_d_in = torch.from_numpy(phi_d_in).float().to(device)    
    brdf_nps_gt = NPs_model(z[0],z[1],phi_d_in, theta_h_in, theta_d_in)

    conca_in = torch.stack([theta_h_in, theta_d_in, phi_d_in], dim=1)
    
    # forward
    z_mean,z_logvar = model(conca_in)
    #print(torch.mean(torch.abs(z[0] - z_pred[:,:7])))
    brdf_nps_pred = NPs_model(z_mean,z_logvar,phi_d_in, theta_h_in, theta_d_in)
    
    loss = torch.mean(torch.abs(brdf_nps_pred - brdf_nps_gt))

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss
    if iteration % 100 == 0 and we_eval:
        brdf_gt_ = brdf_nps_gt.detach().cpu().numpy()
        brdf_pred_ = brdf_nps_pred.detach().cpu().numpy()
        idx = 90000
        print("Ground truth: ", brdf_gt_[idx],"idx: ",idx)
        print("Prediction: ", brdf_pred_[idx],"idx: ",idx)
        print("Iteration: %d, Loss: %f" % (iteration, loss.item()))
        print(torch.mean(torch.abs(z[0] - z_mean)))
        print(torch.mean(torch.abs(z[1] - z_logvar)))
    if iteration % 1000 == 0:
        torch.save(model.state_dict(), "nps_"+ filename + ".pth")