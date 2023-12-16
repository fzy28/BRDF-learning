import torch
import torch.nn as nn
import torch.nn.functional as F

class NPsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 400), nn.ReLU(),  # hidden - 1
            nn.Linear(400, 400), nn.ReLU(), # hidden - 2
            nn.Linear(400, 400), nn.ReLU(), # hidden - 3
            nn.Linear(400, 400), nn.ReLU(), # hidden - 4
            nn.Linear(400, 400), nn.ReLU(), # hidden - 5
            nn.Linear(400, 400), nn.ReLU(), # hidden - 6
            nn.Linear(400, 3), nn.ReLU())   # hidden - 7
    
    def sampling(self,z_mean,z_log_var, disable_variance:bool=False):
        if disable_variance:
            return z_mean
        else:
            epsilon = torch.randn([z_mean.shape[0], 7], device=z_mean.device)
            z_sample = z_mean + torch.exp(z_log_var / 2) * epsilon
            return z_sample
    
    def dot_mask(self, NoL, NoV):
        s_L = F.relu(torch.sign(NoL))
        s_V = F.relu(torch.sign(NoV))
        return s_L * s_V
    
    def ringmap(self, x_grid):
        x_grid_add = x_grid[:, 0:1] * 2 * torch.pi
        x_grid[:, 0:1] = torch.sin(x_grid_add) * 0.5 + 0.5
        x_grid_add = torch.cos(x_grid_add) * 0.5 + 0.5
        x_grid = torch.concatenate([x_grid_add, x_grid], axis=-1)
        return x_grid
    
    def preprocess(self, phi_d, theta_h, theta_d):
        phi_d = phi_d / (torch.pi)
        theta_h = theta_h / (torch.pi/2)
        theta_d = theta_d / (torch.pi/2)
        x_grid = torch.stack([phi_d, theta_h, theta_d], axis=-1)
        return self.ringmap(x_grid)
    
    def postprocess(self, y_hat):
        mat = y_hat * 0.15
        for _ in range(4):
            mat = torch.exp(mat) - 1.0
        return mat
    
    def forward(self,z_mean,z_log_var, phi_d, theta_h, theta_d):
        # concat x and z_sample
        x = self.preprocess(phi_d, theta_h, theta_d)
        z_sample = self.sampling(z_mean, z_log_var)
        y = torch.cat([x, z_sample], -1)
        y_hat = self.fc(y)
        y_hat = self.postprocess(y_hat)
        return y_hat