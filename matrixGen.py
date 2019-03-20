import pandas as pd
import numpy as np

dict={}
dict['target'] = {}
dict['talk'] = {}
dict['vote'] = {}

with open('001/000.log', 'r') as f:
    while True:
        line = f.readline().split(',')
        if line == ['']:
            break

        if line[1] == 'status':
            dict['target'][line[2]] = line[3]

        elif line[1] == 'talk':
            if int(line[4])-1 in dict['talk']:
                dict['talk'][int(line[4]) - 1].append(line[5])
            else:
                dict['talk'][int(line[4]) - 1] = []
                dict['talk'][int(line[4])-1].append(line[5])

        elif line[1] == 'vote':

            if int(line[2])-1 in dict['vote']:
                dict['vote'][int(line[2]) - 1][int(line[3]) - 1] += 1
            else:
                dict['vote'][int(line[2]) - 1] = [0, 0, 0, 0, 0]
                dict['vote'][int(line[2])-1][int(line[3]) - 1] += 1




    print(dict)
