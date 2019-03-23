import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self,net_params):
        super(Net, self).__init__()
        self.rnn = nn.GRU(batch_first=True,input_size=net_params["n_features"],hidden_size=net_params["hidden_size"])
        self.att_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #Receives as input a  Tensor of shape (J,Steps,N_features)
        i, _ = self.rnn(x)



        # For k,q,v in columnvector, ATT = V softmax(K^T Q)
        i = i.permute(1,0,2)
        #We transpose Q to be of shape (Steps,N_features,J)
        Q = i.permute(0,2,1)
        #Same with V
        V = i.permute(0,2,1)
        K = i

        attention_intermediate = torch.matmul(K,Q)
        attention_weights = self.att_softmax(attention_intermediate)

        attention_out = torch.matmul(V,attention_weights)

        print(attention_out.shape)

        #Need to put back to correct shape









        #K = i






