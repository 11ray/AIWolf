import glob
import os
import sys



res_s = sorted(res)

directory = sys.argv[1]
res = glob.glob(directory+'/*.x.npy')
with open(directory+"/train_file_list","w") as f:
    for i in res_s[:int(len(res_s)*0.8)]:
        print(os.path.abspath(i)[:-6],file=f)

with open(directory+"/test_file_list","w") as f:  
    for i in res_s[int(len(res_s)*0.8):]:
        print(os.path.abspath(i)[:-6],file=f)

