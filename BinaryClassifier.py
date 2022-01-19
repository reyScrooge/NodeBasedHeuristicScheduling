import pickle
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@ignore_warnings(category=ConvergenceWarning)
def training(heuristic, Xtrain, Ytrain):
    X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.1)
    while len(np.unique(y_train)) < 2:
        X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size = 0.1)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    print(heuristic, accuracy_score(y_test, y_predict))

    # to save the trained model
    fileName = heuristic + ".sav"
    joblib.dump(lr, fileName)

    #to load the model
    #loadedModel = joblib.load(fileName)


with open('stats.pickle', 'rb') as handle:
    stats = pickle.load(handle)
nodeIDInMatrix = []
nodeIDs = stats.keys()


# Node info in not saved by the order of node ID, so use a list to record the nodeIDs
nodeIDInMatrix = [i for i in nodeIDs]
nodeIDInMatrix = nodeIDInMatrix[1:]



#Choose 100 nodes
successMatrix = np.zeros((500, 17), dtype = int)
heuristics = ['actconsdiving', 'coefdiving', 'conflictdiving', 'crossover', 'distributiondiving', 'farkasdiving', 'fracdiving',
              'guideddiving', 'linesearchdiving', 'localbranching', 'pscostdiving', 'rens', 'rins', 'mutation', 'dins', 'trustregion',
              'veclendiving']

#Choose random Node
chosenNode = random.choices(nodeIDInMatrix, k=500)

for i in range(0, 500):

    nodeId = chosenNode[i]
    heusOfNode = stats[nodeId]['heuStats'].keys()

    for heu in range(len(heuristics)):
        if heuristics[heu] in heusOfNode:
            successMatrix[i][heu] = 1




Xtrain = []
for i in chosenNode:
    Xtrain.append(stats[i]['features'])
Xtrain = np.array(Xtrain)

for heuIndex in range(len(heuristics)):
    heuristic = heuristics[heuIndex]
    Ytrain = successMatrix[:, heuIndex].T.ravel()
    # print(Ytrain.shape)
    # Ytrain = np.array(Ytrain).reshape(len(Xtrain) ,1).ravel()
    if len(np.unique(Ytrain)) > 1:
        training(heuristic, Xtrain, Ytrain)

    # training()





