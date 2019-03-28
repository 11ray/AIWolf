import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import argparse

from src import model,dataset,arguments

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser)
    args = parser.parse_args()
     
    device = torch.device("cpu")
    if (args.gpu_device_number >= 0):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:" + str(args.gpu_device_number) if use_cuda else "cpu")

    
    print("Training using device: ",device)


    net = model.Net(args).to(device)

    train_dataset = dataset.WerewolfDataset('data/gat2017log15_data/sets/train_file_list')
    train_dataloader = data.DataLoader(train_dataset,num_workers=2)

    validation_dataset = dataset.WerewolfDataset('data/gat2017log15_data/sets/test_file_list')
    validation_dataloader = data.DataLoader(validation_dataset,num_workers=2)

    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)

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


            net_output = net.forward(x)


            net_output = net_output.permute(1,0,2).contiguous().view(-1,args.n_roles)

            loss = F.cross_entropy(net_output, y,reduction='none')

            cost = loss.sum() / x.size()[1]

            # Logging
            epoch_cost += cost.detach().cpu().numpy()

            cost.backward()

            if (index_batch + 1) % args.update_frequency == 0:
                if args.update_frequency > 0 and (index_batch + 1) % (args.update_frequency * args.log_frequency) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}'.format(epoch, index_batch, len(train_dataloader),100. * index_batch / len(train_dataloader), cost.item()))
                    print('Network output: ', net_output[-x.size()[0]:,:].argmax(dim=1).detach().cpu().numpy())
                    print( ", target:      ", y[-x.size()[0]:].cpu().numpy())
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

