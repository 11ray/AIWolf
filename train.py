import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src import model,dataset


use_cuda = torch.cuda.is_available()
#device = torch.device("cpu")
#device = torch.device("cuda:0" if use_cuda else "cpu")
#Only works if you have 2 GPUs
device = torch.device("cuda:1" if use_cuda else "cpu")

UPDATE_FREQUENCY = 32
N_ROLES = 6

net = model.Net({"n_features":22,"hidden_size":64,"linear_hidden_size":32,"n_roles":N_ROLES}).to(device)

train_dataset = dataset.WerewolfDataset('data/gat2017log15_data/sets/train_file_list')
train_dataloader = data.DataLoader(train_dataset,num_workers=2)

validation_dataset = dataset.WerewolfDataset('data/gat2017log15_data/sets/test_file_list')
validation_dataloader = data.DataLoader(validation_dataset,num_workers=2)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    print('Epoch {}/{}'.format(epoch, 100 - 1))
    print('-' * 10)

    #Training
    optimizer.zero_grad()
    epoch_cost = 0
    for index_batch, (x, y, valid) in enumerate(train_dataloader):

        #Remove batch 1 dimension
        x = x.view(x.size()[1],x.size()[2],x.size()[3])
        y = y.view(y.size()[1])
        valid = valid.view(valid.size()[1])

        
        """"
        #Careful with the shape, should have shape num_steps
        #We simply create a uniform growing series from 0 to 1
        loss_scale = torch.from_numpy(np.linspace(0.0,1.0,valid.size()[0],dtype='float32'))
        # Add fake last dimension to use expand
        valid = torch.mul(valid, loss_scale).view(valid.size()[0],-1)
        #Valid is expanded, in blocks of J that hold the same value
        valid = valid.expand(-1,x.size()[0]).contiguous().view(-1)


        #Repeat targets across timesteps
        y = y.repeat(x.size()[1])
        """
        
                
        x, y, valid = x.to(device), y.to(device), valid.to(device)


        #Fix valid somewhere along here

        net_output = net.forward(x)


        net_output = net_output.permute(1,0,2).contiguous().view(-1,N_ROLES)

        loss = F.cross_entropy(net_output, y,reduction='none')

        #Perhaps some sort of divsion should be applied here, /steps?
        cost = loss.sum()

        # Logging
        epoch_cost += cost.detach().cpu().numpy()

        cost.backward()

        if (index_batch + 1) % UPDATE_FREQUENCY == 0:
          optimizer.step()
          optimizer.zero_grad()

    print("Train cost/batch :", epoch_cost / len(train_dataloader))

    # Validation
    correct = 0
    denominator = 0
    for index_batch, (x, y, valid) in enumerate(validation_dataloader):
        #Remove batch 1 dimension
        x = x.view(x.size()[1],x.size()[2],x.size()[3])
        y = y.view(y.size()[1])
        valid = valid.view(valid.size()[1])

        net_output = net.forward(x)
        predicted_roles = net_output[:,-1,:].argmax(dim=1)

        denominator+= predicted_roles.numel()
        correct += (predicted_roles == y).float().sum().numpy()[0]

    print("Validation accuracy :", correct / denominator)

