import glob
import os
res = glob.glob('../*.x.npy')

res_s = sorted(res)

with open("train_file_list","w") as f:
    for i in res_s[:int(len(res_s)*0.8)]:
        print(os.path.abspath(i)[:-6],file=f)

with open("test_file_list","w") as f:  
    for i in res_s[int(len(res_s)*0.8):]:
        print(os.path.abspath(i)[:-6],file=f)

