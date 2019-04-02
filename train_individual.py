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

  if args.loss_weights == "inverse_frequency":
    l = [5, 5, 4, 2, 4, 5, 4, 4, 4, 4, 3, 4, 4, 0, 1]
    counts = np.array([1, 1, 1, 1, 8, 3], dtype='float32')
    loss_weights = 1.0 / counts
    #loss_weights = np.array([0.5, 2.0], dtype='float32')
  else:
    loss_weights = np.ones(args.n_roles, dtype='float32')

  weights_t = torch.from_numpy(loss_weights).to(device)
  net = individual_model.Net(args).to(device)

  train_dataset = individual_dataset.IndividualWerewolfDataset(args.train_file_list, args)
  train_dataloader = data.DataLoader(train_dataset, num_workers=6,batch_size=64,collate_fn=individual_dataset.my_collate)

  validation_dataset = individual_dataset.IndividualWerewolfDataset(args.validation_file_list, args)
  validation_dataloader = data.DataLoader(validation_dataset, num_workers=6,batch_size=64,collate_fn=individual_dataset.my_collate)

  optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.0)

  for epoch in range(10):
    print('Epoch {}/{}'.format(epoch, 100 - 1))
    print('-' * 10)

    # Training
    optimizer.zero_grad()
    epoch_cost = 0
    for index_batch, (x, y) in enumerate(train_dataloader):

      x, y = x.to(device), y.to(device)

      net_output = net.forward(x, device)

      net_output = net_output[:,-1,:]

      loss = F.cross_entropy(net_output, y, reduction='mean',weight=weights_t)

      cost = loss / args.update_frequency

      # Logging
      epoch_cost += cost.detach().cpu().numpy()

      cost.backward()

      if (index_batch + 1) % args.update_frequency == 0:
        if args.update_frequency > 0 and (index_batch + 1) % (args.update_frequency * args.log_frequency) == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}'.format(epoch, index_batch, len(train_dataloader),
                                                                         100. * index_batch / len(train_dataloader),
                                                                         cost.item()))
          #print('Network output: ', net_output[-x.size()[0]:, :].argmax(dim=1).detach().cpu().numpy())
          #print(', target:       ', y[-x.size()[0]:].cpu().numpy())
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

      predicted_l.extend(predicted_roles.detach().cpu().numpy().tolist())
      true_l.extend(y.detach().cpu().numpy().tolist())

    print(compute_random_accuracy.compute_2class_stats(np.array(true_l).flatten(), np.array(predicted_l).flatten()))



