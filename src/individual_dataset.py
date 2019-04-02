import torch
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import numpy as np


def my_collate(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    X = nn.utils.rnn.pad_sequence(xs,batch_first=True)
    Y = torch.stack(ys,0)
    return [X, Y]

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

        x = torch.from_numpy(np.load(basename+".x.npy")[row-1]).float()

        if self.args.reduce_classes:
            tmp = np.load(basename + ".y.npy")[row-1]
            y = torch.from_numpy(np.array([self.compress_classes(xi) for xi in tmp])).long()

        else:
            y = torch.tensor(np.load(basename + ".y.npy")[row-1]).long()


        return (x,y)