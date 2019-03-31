import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,net_params):
        super(Net, self).__init__()
        self.rnn = nn.GRU(batch_first=True,input_size=net_params.event_vector_size,hidden_size=net_params.rnn_hidden_size)
        self.att_softmax = nn.Softmax(dim=1)
        self.att_denom = torch.sqrt(torch.tensor(net_params.rnn_hidden_size, dtype=torch.float32))
        self.linear1 = nn.Linear(net_params.rnn_hidden_size, net_params.classifier_hidden_size)

        if net_params.decoder_type == "concat_linear":
            self.output_layer = nn.Linear(net_params.n_players * net_params.rnn_hidden_size + net_params.n_players , net_params.n_roles,bias=False)
        else:
            self.output_layer = nn.Linear(net_params.classifier_hidden_size, net_params.n_roles,bias=False)

        self.args = net_params

    def forward(self, x,device):
        # Receives as input a  Tensor of shape (J,Steps,N_features)
        # If it instead was (1,...), you can do x.view(x.size()[1],x.size()[2],x.size()[3]) to remov the 1 direction

        # Since we use Batch first, J acts in place of the batch
        i, _ = self.rnn(x)

        if self.args.encoder_type == "rnn_selfattention":
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
            # Attention_out has shape (Steps,N_features,J)
            attention_out = torch.matmul(V,attention_weights)

            #We now move to (J,Steps,N_features) in order to apply the linear layers
            i = attention_out.permute(2,0,1)


        if self.args.decoder_type == "concat_linear":
            i = i.permute(1,0,2)
            # We join the representations of all players, getting (Steps, J x N_features)
            i = i.contiguous().view(i.size()[0],-1)

            #Add fake dimension to repeat, we get (J, Steps, J x N_features)
            i = i.view(1,i.size()[0],i.size()[1])
            i = i.expand(self.args.n_players,-1,-1)


            # J x J one hot
            p = torch.eye(self.args.n_players,self.args.n_players)
            # Repeat across Steps
            p = p.view(1,self.args.n_players,self.args.n_players).expand(x.size()[1],-1,-1).to(device)

            # We switch to (Steps,J, -)  to append the one hots
            i = i.permute(1,0,2)

            i = torch.cat([i,p],dim=2)

            i = i.permute(1,0,2)


        else:
            # First linear
            i = self.linear1(i)
            i = F.relu(i)


        #We expect (J,Steps,N_features) shape out of the decoder
        out = self.output_layer(i)



        return out








