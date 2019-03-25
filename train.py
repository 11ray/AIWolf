import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from src import model


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


net = model.Net({"n_features":512,"hidden_size":128,"linear_hidden_size":32,"n_roles":7}).to(device)


exp_steps = 20
exp_players = 15

X = torch.randn((1,exp_players,exp_steps,512))

Y = torch.randint(1,7,(1,exp_players))

VALID_LOSS = torch.round(torch.rand((1,exp_steps)))



train_dataset = data.TensorDataset(X,Y,VALID_LOSS)
train_dataloader = data.DataLoader(train_dataset)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    print('Epoch {}/{}'.format(epoch, 100 - 1))
    print('-' * 10)
    # Training


    running_loss = 0.0
    running_corrects = 0


    for x, y, valid in train_dataloader:

        x = x.view(x.size()[1],x.size()[2],x.size()[3])
        y = y.view(y.size()[1])
        valid = valid.view(valid.size()[1],-1)


        #Valid is expanded, in blocks of J that hold the same value
        valid = valid.expand(-1,exp_players).contiguous().view(-1)

        #Repeat targets across timesteps
        y = y.repeat(exp_steps)

        x, y, valid = x.to(device), y.to(device), valid.to(device)

        #Fix valid somewhere along here

        optimizer.zero_grad()
        net_output = net.forward(x)


        net_output = net_output.permute(1,0,2).contiguous().view(-1,7)

        loss = F.cross_entropy(net_output, y,reduction='none')

        cost = loss.sum()

        print(cost.double())

        cost.backward()
        optimizer.step()




