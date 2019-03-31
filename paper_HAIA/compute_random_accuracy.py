import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report

def compute_baselines_15():
    roles = ["VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER","VILLAGER",\
     "SEER","MEDIUM","BODYGUARD","WEREWOLF","WEREWOLF","WEREWOLF","POSSESSED"]


    n_repetitions = 10000

    predicted_l = []
    true_l = []

    correct = 0
    for _ in range(n_repetitions):
        true = np.random.permutation(roles)
        true_l.append(true)
        predicted = np.random.permutation(roles)
        predicted_l.append(predicted)
        correct += accuracy_score(true,predicted,normalize=False)

    print("Random guess baseline (constrained):")
    print(correct / (n_repetitions*15))
    print("F1 macro:")
    print(f1_score(np.array(true_l).flatten(),np.array(predicted_l).flatten(),average='macro'))



    correct = 0
    for _ in range(n_repetitions):
        true = np.random.permutation(roles)
        predicted = np.array(["VILLAGER"]*15)
        correct += accuracy_score(true,predicted,normalize=False)

    print("Majority class baseline accuracy:")
    #This is simply be 8/15, but it gives peace of mind to simulate it and obtain the right result
    print(correct / (n_repetitions*15))




def compute_2class_stats(true,predicted):
    acc = accuracy_score(true, predicted, normalize=True)


    print("Acc:",acc)
    print("Precision, Recall, F1 (Micro): ",precision_recall_fscore_support(true,predicted,average='micro'))
    print("Precision, Recall, F1 (Macro): ", precision_recall_fscore_support(true, predicted, average='macro'))
    print(classification_report(true,predicted))

def compute_baseline_2():
    print('Random')
    roles = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
    n_repetitions = 10000

    predicted_l = []
    true_l = []
    for _ in range(n_repetitions):
        true = np.random.permutation(roles)
        true_l.append(true)
        predicted = np.random.permutation(roles)
        predicted_l.append(predicted)

    compute_2class_stats(np.array(true_l).flatten(),np.array(predicted_l).flatten())

    #print('Majority class')
    #compute_2class_stats(roles,[0]*15)



