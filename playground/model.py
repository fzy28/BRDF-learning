import torch

N_NEURONS = 128
# architecture
class NN(torch.nn.Module):
    def __init__(self,input_size=3, output_size=3):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size,N_NEURONS)  
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear6 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.output  = torch.nn.Linear(N_NEURONS, output_size) 
        self.activation = torch.nn.SiLU() # More smooth, you may use torch.nn.ReLU() for more sparsity
        self.f_act = torch.nn.ReLU()
    def forward(self, array):

        res = array
        res = self.activation(self.linear1(res))
        res = self.activation(self.linear2(res))
        res = self.activation(self.linear3(res))
        res = self.activation(self.linear4(res))
        res = self.activation(self.linear5(res))
        res = self.activation(self.linear6(res))
        res = self.activation(self.output(res))
        return res
