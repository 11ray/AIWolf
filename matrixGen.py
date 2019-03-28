import pandas as pd
import numpy as np
import re
from sklearn import preprocessing

def generate_matrix(file, name, test=False):
    dict={}
    dict['target'] = {}
    day = 0
    try:
        with open(file, 'r') as f:
            while True:
                line = f.readline().split(',')
                if line == ['']:
                    break
                #print(line)

                # Genero nuevas claves para un nuevo dia de la partida
                if day != int(line[0]):
                    day = int(line[0])
                    dict[day] = {}
                    dict[day]['talk'] = {}
                    dict [day]['vote'] = {}
                    dict [day]['dead'] = {}

                # Almaceno los roles de cada agente (id)
                if line[1] == 'status':
                    dict['target'][line[2]] = line[3]
                    if line[4] == 'DEAD':
                        dict[day]['dead'][int(line[2])] = True

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

            # accion[dead, vote, talk] detalle accion[Vote, comeout, divine, skip/over] target del mensaje[0,0,1,0,0]
            num_players = len(dict['target'].items())
            game = [[] for i in range(num_players)]
            mask = []
            #print(game)
            target = []
            in_dia = 0
            # clave-dia/target valor-tipo_accion{id}
            for key, value in dict.items():
                if key == 'target':
                    for k, v in dict['target'].items():
                        target.append(v)
                else:

                    actions_day = 0
                    # clave-tipo_accion valor-id{sucesos}
                    for k, v in dict[key].items():
                        if k == 'talk':

                            day = []
                            # clave-id valor-[sucesos]
                            for k1, v1 in dict[key][k].items():
                                actions_day = len(v1)
                                acc_dia = []

                                # a -> mensaje
                                for a in v1:

                                    acc_dias = []
                                    accion = [0,0,0]
                                    detalle = [0,0,0,0]
                                    msg_target = [0]*num_players

                                    accion[2] = 1
                                    msg = re.split(' |\[|\]', a)
                                    if msg[0] == 'VOTE':
                                        detalle[0] = 1
                                        msg_target[int(msg[2]) - 1] = 1

                                    elif msg[0] == 'DIVINED':
                                        detalle[2] = 1
                                        msg_target[int(msg[2]) - 1] = 1

                                    elif msg[0] == 'COMINGOUT':
                                        detalle[1] = 1
                                        msg_target[int(msg[2]) - 1] = 1

                                    else:
                                        detalle[3] = 1

                                    acc_dias.extend(accion)
                                    acc_dias.extend(detalle)
                                    acc_dias.extend(msg_target)
                                    acc_dia.append(acc_dias)
                                #print('Acciones del agente',k1,':', len(acc_dia))
                                day.append([acc_dia, k1])


                            flag = 0
                            for p in day:
                                #print(p[1])
                                game[p[1]-1].extend(p[0])
                                for action in p[0]:
                                    if flag == 0:
                                        mask.append(0)
                                flag = 1
                                #game[p[1]-1].extend(p[0])
                                #print(game[p[1]-1])


                        elif k == 'vote':

                            day = []

                            # clave-id valor-[sucesos]
                            for k1, v1 in dict[key][k].items():
                                vot_dia=[]
                                accion = [0, 0, 0]
                                detalle = [0, 0, 0, 0]
                                msg_target = [0] * num_players
                                accion[1] = 1
                                msg_target[v1 - 1] = 1

                                vot_dia.extend(accion)
                                vot_dia.extend(detalle)
                                vot_dia.extend(msg_target)
                                day.append([vot_dia, k1])

                            #print('Votaciones del dia:', day)

                            flag = 0
                            for p in day:
                                #print(p[1])
                                game[p[1]-1].extend([p[0]])
                                if flag == 0:
                                    mask.append(1)
                                flag = 1
                                #print(game[p[1]-1])

                        elif k == 'dead' and key != len(dict.items())-1:
                            day = []
                            for k1, v1 in dict[key][k].items():
                                acc_dia = []
                                accion = [0, 0, 0]
                                detalle = [0, 0, 0, 0]
                                msg_target = [0] * num_players
                                accion[0] = 1

                                acc_dia.extend(accion)
                                acc_dia.extend(detalle)
                                acc_dia.extend(msg_target)

                                day.append([acc_dia, k1])
                            for p in day:
                                real = []
                                for j in range(actions_day+1):
                                    real.append(p[0])


                                game[p[1]-1].extend(real)
                    in_dia += 1

            tgt = []
            enc = preprocessing.OrdinalEncoder()
            X = np.array(target).reshape(-1, 1)
            enc.fit(X)
            for role in X:
                va = np.array([role])
                tgt.append(enc.transform(va))
            target = []
            for elem in tgt:
                target.append(int(elem[0][0]))

            if test:

                print("Players:", len(game))
                print("Steps:", len(game[0]))


                print(np.array(target))
                print(np.array(game))
                print(np.array(mask))
                print(dict)

            else:

                np.save('data/gat2017log05_data/' + name + '.x', np.array(game))
                np.save('data/gat2017log05_data/' + name + '.y', np.array(target))
                np.save('data/gat2017log05_data/' + name + '.valid', np.array(mask))
    except:
        print("error")
if __name__ == "__main__":
    test = False

    for i in range(1000):
        for j in range(100):



            if len(str(i)) == 1:
                f = '00'+str(i)
            elif len(str(i)) == 2:
                f = '0' + str(i)
            else:
                f = str(i)

            if len(str(j)) == 1:
                g = '00'+str(j)
            elif len(str(j)) == 2:
                g = '0' + str(j)
            else:
                g = str(j)

            ubi_file = 'gat2017log05/'+f+'/'+g+'.log'

            print("Obteniendo matriz a partir del fichero: ", ubi_file, ".....")

            if test:
                nombre = str(f) + '_' + str(g)
                generate_matrix('000.log', nombre, test)
                break

            nombre = str(f)+'_'+str(g)
            generate_matrix(ubi_file, nombre, test)

            print("Matriz ", nombre, " guardada")

        if test:
            break
