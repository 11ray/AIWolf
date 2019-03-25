import torch
from src import model


net = model.Net({"n_features":512,"hidden_size":128,"linear_hidden_size":32,"n_roles":7})


x = torch.randn((1,15,20,512))

net.forward(x.view(x.size()[1],x.size()[2],x.size()[3]))