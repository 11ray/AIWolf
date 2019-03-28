import torch
import torch.utils.data as data
import pandas as pd
import numpy as np

class WerewolfDataset(data.Dataset):
  """Werewolf dataset."""

  def __init__(self, name_filelist ):
    self.name_dataframe = pd.read_csv(name_filelist)


  def __len__(self):
    return len(self.name_dataframe)

  def __getitem__(self, idx):
    basename = self.name_dataframe.iloc[idx][0]


    x = torch.from_numpy(np.load(basename+".x.npy")).float()
    y = torch.from_numpy(np.load(basename + ".y.npy")).long()
    valid = torch.from_numpy(np.load(basename + ".valid.npy")).float()


    return (x,y,valid)