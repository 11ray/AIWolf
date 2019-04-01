import pandas as pd
import os

def get_id(lines,n_players):
    r = []
    for line in lines[:n_players]:
        #_, _, player_id, _, _, agent_id
        r.append(line.split(","))
        
        
    #Player_id, agent_id
    s = zip(map( lambda x:  x[2],r), map(lambda x:  x[5].rstrip(),r))
    
    return s
        

def generate_player_game_list(basename,agent_id,values,split=False):
    if split:
        with open(basename+"/" + agent_id + ".train.set" ,'w') as f:
            for game,id in values[:int(len(values)*0.8)]:
                print( ",".join([os.path.abspath(basename+game),id]),file=f)

        with open(basename+"/" + agent_id + ".test.set" ,'w') as f:
            for game,id in values[int(len(values)*0.8):]:
                print( ",".join([os.path.abspath(basename+game),id]),file=f)

    else:
        with open(basename+"/" + agent_id + ".set" ,'w') as f:
            for game,id in values:
                print( ",".join([os.path.abspath(basename+game),id]),file=f)
        
if __name__ == "__main__":
    agent_dataset_dict = {}
    for i in range(1000):
        for j in range(100):
            try:
                with open('data/gat2017log15/' + str(i).zfill(3) +'/' + str(j).zfill(3) + '.log') as f:
                    r = get_id(f.readlines(),15)
                    for player_id, agent_id in r:
                        d = agent_dataset_dict.get(agent_id)
                        if d is None:
                            a = []
                            a.append((str(i).zfill(3) +'_' + str(j).zfill(3),player_id))
                            agent_dataset_dict[agent_id] = a
                        else:
                            d.append((str(i).zfill(3) +'_' + str(j).zfill(3),player_id))


            except Exception as e:
                print("Problem generating ",str(i).zfill(3),'_',str(j).zfill(3))
                print(e)

    for agent_id in list(agent_dataset_dict.keys()):
        if not agent_id.startswith('Dummy'):
            generate_player_game_list('data/gat2017log15_data/',agent_id,agent_dataset_dict[agent_id],split=True)

