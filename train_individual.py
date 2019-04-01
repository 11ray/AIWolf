import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import argparse

from src import individual_model, individual_dataset, arguments
from paper_HAIA import compute_random_accuracy

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  arguments.add_arguments(parser)
  args = parser.parse_args()

  device = torch.device("cpu")
  if (args.gpu_device_number >= 0):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu_device_number) if use_cuda else "cpu")

  print("Training using device: ", device)


  net = individual_model.Net(args).to(device)

  train_dataset = individual_dataset.IndividualWerewolfDataset(args.train_file_list, args)
  train_dataloader = data.DataLoader(train_dataset, num_workers=3,batch_size=1)

  validation_dataset = individual_dataset.IndividualWerewolfDataset(args.validation_file_list, args)
  validation_dataloader = data.DataLoader(validation_dataset, num_workers=3,batch_size=1)

  optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.0)

  for epoch in range(100):
    print('Epoch {}/{}'.format(epoch, 100 - 1))
    print('-' * 10)

    # Training
    optimizer.zero_grad()
    epoch_cost = 0
    for index_batch, (x, y) in enumerate(train_dataloader):


      x, y = x.to(device), y.to(device)

      net_output = net.forward(x, device)

      net_output = net_output[:,-1,:]

      print(net_output)

      loss = F.cross_entropy(net_output, y, reduction='none')

      cost = loss.sum() / args.update_frequency

      # Logging
      epoch_cost += cost.detach().cpu().numpy()

      cost.backward()

      if (index_batch + 1) % args.update_frequency == 0:
        if args.update_frequency > 0 and (index_batch + 1) % (args.update_frequency * args.log_frequency) == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}'.format(epoch, index_batch, len(train_dataloader),
                                                                         100. * index_batch / len(train_dataloader),
                                                                         cost.item()))
          print('Network output: ', net_output[-x.size()[0]:, :].argmax(dim=1).detach().cpu().numpy())
          print(', target:       ', y[-x.size()[0]:].cpu().numpy())
        optimizer.step()
        optimizer.zero_grad()

    print("Train cost/batch :", epoch_cost / len(train_dataloader))

    # Validation
    predicted_l = []
    true_l = []

    for index_batch, (x, y,) in enumerate(validation_dataloader):


      x, y = x.to(device), y.to(device)

      net_output = net.forward(x, device)
      predicted_roles = net_output[:, -1, :].argmax(dim=1)

      predicted_l.append(predicted_roles.detach().cpu().numpy())
      true_l.append(y[-args.n_players:].detach().cpu().numpy())

    print(predicted_l)
    print(compute_random_accuracy.compute_2class_stats(np.array(true_l).flatten(), np.array(predicted_l).flatten()))



