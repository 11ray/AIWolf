import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,net_params):
        super(Net, self).__init__()
        self.rnn = nn.GRU(batch_first=True,input_size=net_params["n_features"],hidden_size=net_params["hidden_size"])
        self.att_softmax = nn.Softmax(dim=1)
        self.att_denom = torch.sqrt(torch.tensor(net_params["hidden_size"], dtype=torch.float32))
        self.linear1 = nn.Linear(net_params["hidden_size"], net_params["linear_hidden_size"])
        self.linear2 = nn.Linear(net_params["linear_hidden_size"],net_params["n_roles"])

    def forward(self, x):
        # Receives as input a  Tensor of shape (J,Steps,N_features)
        # If it instead was (1,...), you can do x.view(x.size()[1],x.size()[2],x.size()[3]) to remov the 1 direction

        # Since we use Batch first, J acts in place of the batch
        i, _ = self.rnn(x)

        # Now we exchange J and steps, resulting in so (Steps,J,N_features)
        # At the output of each step, we wish to apply self attention
        i = i.permute(1,0,2)

        # For k,q,v in columnvector, ATT = V softmax(K^T Q)
        #We transpose Q to be of shape (Steps,N_features,J)
        Q = i.permute(0,2,1)
        #Same with V
        V = i.permute(0,2,1)
        #K is already in the right shape
        K = i

        # Self attention
        attention_intermediate = torch.div( torch.matmul(K,Q), self.att_denom)
        attention_weights = self.att_softmax(attention_intermediate)
        attention_out = torch.matmul(V,attention_weights)


        #We now move to (Steps,J,N_features) in order to apply the linear layers
        i = attention_out.permute(0,2,1)

        # First linear
        i = self.linear1(i)
        i = F.relu(i)

        i = self.linear2(i)


        #Back to (J,Steps,N_features)
        out = i.permute(1,0,2)

        return out








