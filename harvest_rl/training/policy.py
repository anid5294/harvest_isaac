import torch
import numpy as np
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, drone_state, drone_state_input_size=9):
        super().__init__()
        self.drone_state = drone_state
        self.drone_state_input_size = drone_state_input_size
        self.mlp = nn.Sequential(
            nn.Linear(in_features=drone_state_input_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=4),
        )
        
        return
    
    def forward(self, curr_drone_state):
        # batch_size = curr_drone_state.shape[0]
        self.drone_state = torch.from_numpy(curr_drone_state)
        # predicting the two means of our two rotor motor values 
        gaussian_params = self.mlp(self.drone_state)
        gaussian_params_exp = torch.zeros(gaussian_params.shape)
        gaussian_params_exp[2:] = torch.exp(gaussian_params[2:])
        gaussian_params_exp[:2] = gaussian_params[:2]
        return gaussian_params_exp
    
    
class CriticNetwork(nn.Module):
    def __init__(self, drone_state, drone_state_input_size=9):
        super().__init__()
        self.drone_state = drone_state
        self.mlp = nn.Sequential(
            nn.Linear(in_features=drone_state_input_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
    
    def forward(self, curr_drone_state):
        return self.mlp(torch.from_numpy(curr_drone_state))