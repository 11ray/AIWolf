import copy

agents = ['AITKN','carlo','cash','daisyo','JuN1Ro','m_cre','megumish','tori','TOT','wasabi']

basename='data/gat2017log15_data/'
for agent in agents:
    with open(basename+agent+".train_all_1000.set", 'w') as outfile:
        rest_of_agents = copy.deepcopy(agents)
        rest_of_agents.remove(agent)
        for a in rest_of_agents:
            with open(basename+a+'.train_1000.set') as infile:
                for line in infile:
                    outfile.write(line)
