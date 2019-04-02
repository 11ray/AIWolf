import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import argparse

from src import model,dataset,arguments
from paper_HAIA import compute_random_accuracy





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arguments.add_arguments(parser)
    args = parser.parse_args()




     
    device = torch.device("cpu")
    if (args.gpu_device_number >= 0):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:" + str(args.gpu_device_number) if use_cuda else "cpu")

    
    print("Training using device: ",device)

    if args.loss_weights =="inverse_frequency":
        l = [5, 5, 4, 2, 4, 5, 4, 4, 4, 4, 3, 4, 4, 0, 1]
        #counts = np.array([1, 1, 1, 1, 8, 3], dtype='float32')
        counts = np.array([12, 3], dtype='float32')
        #loss_weights = 1.0 / counts
        loss_weights = np.array([0.5,2.0], dtype='float32')

    else:
        loss_weights = np.ones(args.n_roles, dtype='float32')


    weights_t = torch.from_numpy(loss_weights).to(device)

    net = model.Net(args).to(device)

    train_dataset = dataset.WerewolfDataset(args.train_file_list,args)
    train_dataloader = data.DataLoader(train_dataset,num_workers=2)

    validation_dataset = dataset.WerewolfDataset(args.validation_file_list,args)
    validation_dataloader = data.DataLoader(validation_dataset,num_workers=2)

    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.001, momentum=0.0)

    for epoch in range(1):
        print('Epoch {}/{}'.format(epoch, 1 - 1))
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


            net_output = net.forward(x,device)


            net_output = net_output.permute(1,0,2).contiguous().view(-1,args.n_roles)


            loss = torch.mul(F.cross_entropy(net_output, y,reduction='none',weight=weights_t),valid)

            cost = loss.sum() / args.update_frequency

            if args.loss_scale != "last_step_only":
                cost = cost / x.size()[1]

            # Logging
            epoch_cost += cost.detach().cpu().numpy()

            cost.backward()

            if (index_batch + 1) % args.update_frequency == 0:
                if args.update_frequency > 0 and (index_batch + 1) % (args.update_frequency * args.log_frequency) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}'.format(epoch, index_batch, len(train_dataloader),100. * index_batch / len(train_dataloader), cost.item()))
                    print('Network output: ', net_output[-x.size()[0]:,:].argmax(dim=1).detach().cpu().numpy())
                    print(', target:       ', y[-x.size()[0]:].cpu().numpy())
                optimizer.step()
                optimizer.zero_grad()


        print("Train cost/batch :", epoch_cost / len(train_dataloader))

        # Validation
        predicted_l = []
        true_l = []


        for index_batch, (x, y, valid) in enumerate(validation_dataloader):
            #Remove batch 1 dimension
            x = x.view(x.size()[1],x.size()[2],x.size()[3])
            y = y.view(y.size()[1])
            #valid = valid.view(valid.size()[1])

            x, y  = x.to(device), y.to(device)

            net_output = net.forward(x,device)
            predicted_roles = net_output[:,-1,:].argmax(dim=1)

            predicted_l.append(predicted_roles.detach().cpu().numpy())
            true_l.append(y[-args.n_players:].detach().cpu().numpy())


        print(compute_random_accuracy.compute_2class_stats(np.array(true_l).flatten(),np.array(predicted_l).flatten()))




