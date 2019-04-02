import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report

number_features = 17
counts = np.array([1, 1, 1, 1, 8, 3], dtype='float32')
loss_weights = 1.0 / counts

# Features [ESTIMATE, DIVINATION, GUARD, VOTE, ATTACK, DIVINED, IDENTIFIED, GUARDED, VOTED, ATTACKED, SKIP/OVER,
#  COMING_OUT_0, COMING_OUT_1, COMING_OUT_2, COMING_OUT_3, COMING_OUT_4, COMING_OUT_5]
def get_which_feature_is(x):
  # x is (1, vector_event_size) or vector_event_size
  detalle = x[3:15]
  role = x[30:36]
  if detalle.sum() > 0:
    detalle_int = np.argmax(detalle)
    if detalle_int !=1:
      if detalle_int > 1:
        return detalle_int - 1
      else:
        return detalle_int
    else:
      feat_base = 11
      return feat_base + np.argmax(role)

  else:
    return None


def extract_features(X):
  # X is (Seq_len, vector_event_size)
  result_arr = np.zeros((1, number_features))
  for i in range(X.shape[0]):
    feat = get_which_feature_is(X[i][:])

    if feat != None:
      result_arr[0][feat] += 1
  return result_arr



def get_X_Y(file):
  with open(file) as f:
    X = []
    Y = []
    for line in f:
      (file,player_id) = line.split(",")
      x = np.load(file+".x.npy")[int(player_id)-1][:][:]
      y = np.load(file+".y.npy")[int(player_id)-1]
      features = extract_features(x)



      X.append(features)
      Y.append(y)

  return np.array(X).reshape((-1,number_features)), np.array(Y)




X_train, Y_train = get_X_Y('../cash.train_1000.set')
X_test, Y_test = get_X_Y('../cash.test_1000.set')
X_train = np.clip(X_train,0,1)
X_test = np.clip(X_test,0,1)



#clf = BernoulliNB(alpha=0,fit_prior=True)
clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_hat = clf.predict(X_test)

print(classification_report(y_true=Y_test,y_pred=Y_hat))

