import torch
import torch.nn as nn
import numpy as np


# https://github.com/openai/evolution-strategies-starter/blob/951f19986921135739633fb23e55b2075f66c2e6/es_distributed/tf_util.py#L109

def normc_initializer(weight, std=0.1):
    out = np.random.randn(*weight.size()).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    weight.data = torch.from_numpy(out)


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, low=-1, high=1):
        super().__init__()
        self.low = low
        self.high = high
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_size)

        normc_initializer(self.l1.weight, std=1.0)
        normc_initializer(self.l2.weight, std=1.0)
        normc_initializer(self.l3.weight, std=0.01)
        self.l1.bias.data.fill_(0.0)
        self.l2.bias.data.fill_(0.0)
        self.l3.bias.data.fill_(0.0)

        self.state_mean = np.zeros(state_size)
        self.state_std = np.ones(state_size)
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x
    
    def act(self, state, noise_std=0.01):
        state = ((state - self.state_mean) / self.state_std).clip(-5.0, 5.0)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.forward(state)
            action += torch.randn_like(action) * noise_std
        return action.numpy().ravel().clip(self.low, self.high)
    
    def set_state_stat(self, mean, std):
        self.state_mean = mean
        self.state_std = std
    
    @property
    def num_params(self):
        return sum([np.prod(params.size()) for params in self.state_dict().values()])

    def get_params(self):
        return torch.cat([params.flatten() for params in self.state_dict().values()])
    
    def set_params(self, all_params):
        all_params = torch.FloatTensor(all_params)
        state_dict = dict()
        for key, params in self.state_dict().items():
            size = params.size()
            state_dict[key] = all_params[:np.prod(size)].view(*size)
            all_params = all_params[np.prod(size):]
        self.load_state_dict(state_dict)
    
    def add_params(self, diff_params):
        diff_params = torch.FloatTensor(diff_params)
        state_dict = dict()
        for key, params in self.state_dict().items():
            size = params.size()
            state_dict[key] = params + diff_params[:np.prod(size)].view(*size)
            diff_params = diff_params[np.prod(size):]
        self.load_state_dict(state_dict)
