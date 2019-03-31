import numpy as np
from sklearn.metrics import accuracy_score


roles = ["VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER",\
 "SEER","MEDIUM","BODYGUARD","WEREWOLF","WEREWOLF","WEREWOLF","POSSESSED"]
 
 
n_repetitions = 1000000



correct = 0
for _ in range(n_repetitions):
    true = np.random.permutation(roles)
    predicted = np.random.permutation(roles)
    correct += accuracy_score(true,predicted,normalize=False)

print("Random guess baseline (constrained):")
print(correct / (n_repetitions*15))




correct = 0
for _ in range(n_repetitions):
    true = np.random.permutation(roles)
    predicted = np.array(["VILLAGER"]*15)
    correct += accuracy_score(true,predicted,normalize=False)

print("Majority class baseline accuracy:")
#This is simply be 8/15, but it gives peace of mind to simulate it and obtain the right result
print(correct / (n_repetitions*15))
