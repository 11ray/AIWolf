import torch
import numpy as np

n_samples = 1000
exp_steps = 20
exp_players = 15
n_roles = 7
n_features = 512


for i in range(n_samples):
  X = torch.randn((exp_players,exp_steps,n_features)).numpy()
  Y = torch.randint(1,n_roles,(exp_players,)).numpy()
  VALID_LOSS = torch.round(torch.rand((exp_steps))).numpy()

  np.save('data/synthetic/synthetic_train/'+str(i)+'.x',X)
  np.save('data/synthetic/synthetic_train/'+str(i)+'.y',Y)
  np.save('data/synthetic/synthetic_train/'+str(i)+'.valid',VALID_LOSS)


for i in range(100):
  X = torch.randn((exp_players,exp_steps,n_features)).numpy()
  Y = torch.randint(1,n_roles,(exp_players,)).numpy()
  VALID_LOSS = torch.round(torch.rand((exp_steps))).numpy()

  np.save('data/synthetic/synthetic_valid/'+str(i)+'.x',X)
  np.save('data/synthetic/synthetic_valid/'+str(i)+'.y',Y)
  np.save('data/synthetic/synthetic_valid/'+str(i)+'.valid',VALID_LOSS)
