import pandas as pd
import numpy as np

dict={}
dict['target'] = {}

day = 0
with open('001/000.log', 'r') as f:
    while True:
        line = f.readline().split(',')
        if line == ['']:
            break

        # Almaceno los roles de cada agente (id)
        if line[1] == 'status':
            dict['target'][line[2]] = line[3]

        # Genero nuevas claves para un nuevo dia de la partida
        if day != int(line[0]):
            day = int(line[0])
            dict[day] = {}
            dict[day]['talk'] = {}
            dict [day]['vote'] = {}

        # Almaceno la sucesión de mensajes de cada agente para cada dia
        if line[1] == 'talk' and day!=0:
            if int(line[4]) in dict[day]['talk']:
                dict[day]['talk'][int(line[4])].append(line[5])
            else:
                dict[day]['talk'][int(line[4])] = []
                dict[day]['talk'][int(line[4])].append(line[5])

        # Almaceno la votación realizada al finalizar cada dia
        elif line[1] == 'vote' and day != 0:
            dict[day]['vote'][int(line[2])] = int(line[3])





    print(dict)
