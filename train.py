import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src import model


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
UPDATE_FREQUENCY = 100

net = model.Net({"n_features":512,"hidden_size":128,"linear_hidden_size":32,"n_roles":7}).to(device)


exp_steps = 20
exp_players = 15

X = torch.randn((1000,exp_players,exp_steps,512))

Y = torch.randint(1,7,(1000,exp_players))

VALID_LOSS = torch.round(torch.rand((1000,exp_steps)))



train_dataset = data.TensorDataset(X,Y,VALID_LOSS)
#Dataloader is an iterable
train_dataloader = data.DataLoader(train_dataset)

optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.0)

for epoch in range(100):
    print('Epoch {}/{}'.format(epoch, 100 - 1))
    print('-' * 10)


    optimizer.zero_grad()

    epoch_cost = 0

    for index_batch, (x, y, valid) in enumerate(train_dataloader):

        #Remove batch 1 dimension
        x = x.view(x.size()[1],x.size()[2],x.size()[3])
        y = y.view(y.size()[1])
        valid = valid.view(valid.size()[1])


        #Careful with the shape, should have shape num_steps
        #We simply create a uniform growing series from 0 to 1
        loss_scale = torch.from_numpy(np.linspace(0.0,1.0,valid.size()[0],dtype='float32'))
        # Add fake last dimension to use expand
        valid = torch.mul(valid, loss_scale).view(valid.size()[0],-1)
        #Valid is expanded, in blocks of J that hold the same value
        valid = valid.expand(-1,exp_players).contiguous().view(-1)



        #Repeat targets across timesteps
        y = y.repeat(exp_steps)

        x, y, valid = x.to(device), y.to(device), valid.to(device)

        #Fix valid somewhere along here

        net_output = net.forward(x)


        net_output = net_output.permute(1,0,2).contiguous().view(-1,7)

        loss = F.cross_entropy(net_output, y,reduction='none')

        #Perhaps some sort of divsion should be applied here, /steps?
        cost = loss.sum()

        # Logging
        epoch_cost += cost.detach().numpy()

        cost.backward()




        if (index_batch + 1) % UPDATE_FREQUENCY == 0:
          optimizer.step()
          optimizer.zero_grad()

    print(epoch_cost)



