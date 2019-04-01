import torch
import torch.utils.data as data
import pandas as pd
import numpy as np

class IndividualWerewolfDataset(data.Dataset):
    """Werewolf dataset."""

    def __init__(self, name_filelist,args):
        self.name_dataframe = pd.read_csv(name_filelist)
        self.args = args

    def compress_classes(self,x):
        if x == 5:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.name_dataframe)

    def __getitem__(self, idx):
        basename = self.name_dataframe.iloc[idx][0]
        row = self.name_dataframe.iloc[idx][1]

        x = torch.from_numpy(np.load(basename+".x.npy")[row]).float()

        if self.args.reduce_classes:
            tmp = np.load(basename + ".y.npy")[row]
            y = torch.from_numpy(np.array([self.compress_classes(xi) for xi in tmp])).long()

        else:
            y = torch.from_numpy(np.array(np.load(basename + ".y.npy")[row])).long()


        return (x,y)