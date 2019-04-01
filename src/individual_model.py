import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,net_params):
        super(Net, self).__init__()
        self.rnn = nn.GRU(batch_first=True,input_size=net_params.event_vector_size,hidden_size=net_params.rnn_hidden_size)
        self.output_layer = nn.Linear(net_params.rnn_hidden_size, net_params.n_roles,bias=False)

        self.args = net_params

    def forward(self, x,device):
        # Receives as input a  Tensor of shape (Batch,Steps,N_features)

        i, _ = self.rnn(x)
        out = self.output_layer(i)

        return out


