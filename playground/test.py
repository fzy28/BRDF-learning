import torch.nn.functional as F
import torch
# Example data (you will replace these with your actual data)
nrays = 4
nlights = 3
pts2c = torch.rand(nrays, 3)  # Random tensor to simulate pts2c

# Normalize V
V = F.normalize(pts2c, dim=-1)  # [nrays, 3]

# Repeat each row of V 'nlights' times
V_repeated = V.unsqueeze(1).repeat(1, nlights, 1)  # [nrays, nlights, 3]
V_reshape = V_repeated.reshape(-1,3)
V_reshape = V_reshape.reshape(nrays, nlights, 3)
print(V_repeated.shape, V_repeated)
print(V_reshape.shape, V_reshape)
