from src import non_sequence_feature_extraction
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report
import sys
import warnings


import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
if not sys.warnoptions:
    warnings.simplefilter("ignore")



counts = np.array([1, 1, 1, 1, 8, 3], dtype='float32')
loss_weights = 1.0 / counts
clf = LogisticRegression()

def train_and_test_classifier(classifier,train_file,test_file,limits=None,average='macro',):
    clf = classifier
    X_train, Y_train = non_sequence_feature_extraction.get_X_Y(train_file)
    X_test, Y_test = non_sequence_feature_extraction.get_X_Y(test_file)
    X_train = np.clip(X_train,0,1)
    X_test = np.clip(X_test,0,1)

    if limits == None:
        clf.fit(X_train, Y_train)
        Y_hat = clf.predict(X_test)
        return f1_score(Y_test,Y_hat,average=average)

    else:
        results = []
        for l in limits:
            clf.fit(X_train[:l], Y_train[:l])
            Y_hat = clf.predict(X_test)
            results.append(f1_score(Y_test,Y_hat,average=average))
        return results

series = []
limits = [10,25,50,100,1000]
average = 'micro'
for agent in ['AITKN','carlo','cash','daisyo','JuN1Ro','m_cre','megumish','tori','TOT','wasabi']:
    print("Agent: ", agent)
    results = train_and_test_classifier(LogisticRegression(), 'data/gat2017log15_data/' + agent + '.train_1000.set',
                                        'data/gat2017log15_data/' + agent + '.test_1000.set',limits,average)
    series.append((agent,results))


li = [str(i) for i in limits]
for s in series:
    plt.plot(li,s[1],'o-',label=s[0])

if average == 'macro':
    baseline = 0.16
else:
    baseline = 0.34
plt.plot(li,[baseline]*len(li),linestyle='dashed',label='Baseline',color='black')
plt.legend()
plt.show()
