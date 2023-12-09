import torch
import torch.nn as nn

N_NEURONS = 64

# Positional Encoder (frequency)
class PositionalEncoder(nn.Module):
    # sine-cosine positional encoder for input points.
    def __init__(self, n_freqs: int):
        super().__init__()
        self.embed_fns = [lambda x: x]
        freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    # forward to positional encoder
    def forward(self, x) -> torch.Tensor:
        # Apply positional encoding to input.
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


# architecture
class NN(torch.nn.Module):
    def __init__(self, input_size=3, output_size=3, freqs=10):
        super().__init__()
        self.pe = PositionalEncoder(freqs)
        pe_size = 3 * (1 + 2 * freqs)
        self.linear1 = torch.nn.Linear(pe_size,N_NEURONS)  
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.output  = torch.nn.Linear(N_NEURONS, output_size) 
        self.activation = torch.nn.SiLU() # More smooth, you may use torch.nn.ReLU() for more sparsity
        self.f_act = torch.nn.ReLU()
    
    def forward(self, array):
        res = array
        res = self.pe(res).detach()
        res = self.activation(self.linear1(res))
        res = self.activation(self.linear2(res))
        res = self.activation(self.linear3(res))
        res = self.activation(self.linear4(res))
        res = self.output(res)
        return res
