'''
Base implementations of the fourier neural operator and U-shaped neural operator. 
Implementations from the neural operator library, https://github.com/neuraloperator/neuraloperator
'''
from neuralop.models import UNO, FNO
import torch.nn as nn

class ushapedno(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(ushapedno, self).__init__()
        self.model = UNO(in_channels = in_channels, out_channels=out_channels, hidden_channels=hidden_channels, n_layers=6, uno_out_channels =  [512,512,512,512,512, 512], uno_n_modes=[[6,6],[4,4],[2,2],[2,2], [4,4],[6,6]], uno_scalings=[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]], channel_mlp_skip='linear')
    def forward(self, inp):
        return self.model(inp)
class fourierno(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(fourierno, self).__init__()
        self.model = FNO(in_channels = in_channels, out_channels=out_channels, hidden_channels=hidden_channels, n_modes=(120,8, 8), n_layers=12)
    def forward(self, inp):
        
        return self.model(inp)