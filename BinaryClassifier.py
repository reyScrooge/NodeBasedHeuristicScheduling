import pickle
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


with open('stats.pickle', 'rb') as handle:
    stats = pickle.load(handle)
nodeIDInMatrix = []
nodeIDs = stats.keys()


# Node info in not saved by the order of node ID, so use a list to record the nodeIDs
nodeIDInMatrix = [i for i in nodeIDs]
nodeIDInMatrix = nodeIDInMatrix[1:]



#Choose 100 nodes
successMatrix = np.zeros((100, 17), dtype = int)
heuristics = ['actconsdiving', 'coefdiving', 'conflictdiving', 'crossover', 'distributiondiving', 'farkasdiving', 'fracdiving',
              'guideddiving', 'linesearchdiving', 'localbranching', 'pscostdiving', 'rens', 'rins', 'mutation', 'dins', 'trustregion',
              'veclendiving']
#Choose random Node
chosenNode = random.choices(nodeIDInMatrix, k=100)

for i in range(0, 100):

    nodeId = chosenNode[i]
    heusOfNode = stats[nodeId]['heuStats'].keys()

    for heu in range(len(heuristics)):
        if heuristics[heu] in heusOfNode:
            successMatrix[i][heu] = 1




Xtrain = []
for i in chosenNode:
    Xtrain.append(stats[i]['features'])
Xtrain = np.array(Xtrain)


@ignore_warnings(category=ConvergenceWarning)
def training():
    for heuIndex in range(len(heuristics)):

        heuristic = heuristics[heuIndex]
        Ytrain = successMatrix[:, heuIndex].T.ravel()
        print(Ytrain.shape)
        # Ytrain = np.array(Ytrain).reshape(len(Xtrain) ,1).ravel()

        lr = LogisticRegression()
        lr.fit(Xtrain, Ytrain)

        # to save the trained model
        fileName = heuristic + ".sav"
        joblib.dump(lr, fileName)

    #to load the model
    #loadedModel = joblib.load(fileName)

training()
