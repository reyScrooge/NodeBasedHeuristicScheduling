import pyscipopt
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class EventHandler(pyscipopt.Eventhdlr):
    """
    A SCIP event handler that records solving stats
    """
    def __init__(self):
        self.stats = {}

    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event):

        event_type = event.getType()
        if (event_type == pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE):
        	print('Node feasible')
        if (event_type == pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE):
        	print('Node infeasible')
        if (event_type == pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED):
        	print('Node branched')

        depth = self.model.getDepth()
        primal = self.model.getPrimalbound()
        print('depth ', depth)
        print("primal ", primal)

        nodeID = self.model.getCurrentNode().getNumber()
        nodeFeatures = []


        # optimality gap
        optimality_gap = self.model.getGap()
        nodeFeatures.append(optimality_gap)

        # infinity gap
        isInf = self.model.isInfinity(optimality_gap)
        nodeFeatures.append(isInf)

        # Root LP value / Global Lower Bound
        nodeFeatures.append(self.model.getDualboundRoot( ) /self.model.getLPObjVal())

        # Root LP value / Global Upper Bound
        nodeFeatures.append(self.model.getDualboundRoot( ) /(self.model.getLPObjVal( ) +optimality_gap))

        # Node Depth
        nodeFeatures.append(self.model.getDepth())

        # Get fractional variables
        fracVars, _, fractionalities, nOfFrac, _, _ = self.model.getLPBranchCands()

        nOfUpLocks = []
        nOfDownLocks = []
        distanceToLPSol = []
        vectorLen = []
        pseudocost = []
        for var in fracVars:

            nOfUpLocks.append(var.getNLocksUp())
            nOfDownLocks.append(var.getNLocksDown())
            distanceToLPSol.append(var.getLPSol( ) -self.model.getDualboundRoot())

            #vector length
            cost = var.getObj()
            col = var.getCol()
            temp = np.array(col.getVals())
            LPSol = var.getLPSol()
            frac = LPSol - np.floor(LPSol)
            if (cost > 0):
                vec_length = ( 1 -frac) * cost / (np.linalg.norm(temp ) +1)
            else:
                vec_length = - frac * cost / (np.linalg.norm(temp) + 1)
            vectorLen.append(vec_length)
            # vectorLen.append(0)
            pseudocost.append(self.model.getVarPseudocost(var))

        # Sum of variables’ LP solution fractionalities / Num. of Fractional Variables
        if nOfFrac == 0:
            nodeFeatures.append(1)
        else:
            nodeFeatures.append(np.sum(fractionalities) / nOfFrac)

        # Num. of Fractional Variable / Num. of Integer Variables
        nInt = self.model.getNIntVars()
        if (nInt - nOfFrac) == 0:
            nodeFeatures.append(1)
        else:
            nodeFeatures.append(nOfFrac / (nInt - nOfFrac))

        # Num. of Active Constraints / Num. of Constraints
        conss = self.model.getConss()
        nOfActiveCons = 0
        for cons in conss:
            if(cons.isActive()):
                nOfActiveCons += 1
        nodeFeatures.append(nOfActiveCons/ len(conss))

        # Node is root
        if(nodeID == 1):
            nodeFeatures.append(True)
        else:
            nodeFeatures.append(False)

        # Root LP value / Node LB
        nodeFeatures.append(self.model.getDualboundRoot() / self.model.getCurrentNode().getLowerbound())
        # Root LP value / Node Estimate

        # Node estimate
        estimate = self.model.getCurrentNode().getEstimate()
        nodeFeatures.append(self.model.getDualboundRoot() / estimate)

        # Number of Up Locks
        if nOfFrac == 0:
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
        else:
            nodeFeatures.append(np.mean(nOfUpLocks))
            nodeFeatures.append(np.min(nOfUpLocks))
            nodeFeatures.append(np.max(nOfUpLocks))
            nodeFeatures.append(np.std(nOfUpLocks))
            nodeFeatures.append(np.median(nOfUpLocks))

        # Number of Down Locks
        if nOfFrac == 0:
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
        else:
            nodeFeatures.append(np.mean(nOfDownLocks))
            nodeFeatures.append(np.min(nOfDownLocks))
            nodeFeatures.append(np.max(nOfDownLocks))
            nodeFeatures.append(np.std(nOfDownLocks))
            nodeFeatures.append(np.median(nOfDownLocks))

        # Normalized Objective Coefﬁcient

        # Distance to root LP solution
        if nOfFrac == 0:
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
        else:
            nodeFeatures.append(np.mean(distanceToLPSol))
            nodeFeatures.append(np.min(distanceToLPSol))
            nodeFeatures.append(np.max(distanceToLPSol))
            nodeFeatures.append(np.std(distanceToLPSol))
            nodeFeatures.append(np.median(distanceToLPSol))

        # Vector Length
        if nOfFrac == 0:
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
        else:
            nodeFeatures.append(np.mean(vectorLen))
            nodeFeatures.append(np.min(vectorLen))
            nodeFeatures.append(np.max(vectorLen))
            nodeFeatures.append(np.std(vectorLen))
            nodeFeatures.append(np.median(vectorLen))

        # Pseudocost score
        if nOfFrac == 0:
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
            nodeFeatures.append(0)
        else:
            nodeFeatures.append(np.mean(pseudocost))
            nodeFeatures.append(np.min(pseudocost))
            nodeFeatures.append(np.max(pseudocost))
            nodeFeatures.append(np.std(pseudocost))
            nodeFeatures.append(np.median(pseudocost))


        self.stats[nodeID] = {}
        self.stats[nodeID]['features'] = nodeFeatures




instance = "ins.lp"


m = pyscipopt.Model()
m.readParams("params.set")
m.setParam('display/verblevel', 0)
m.setParam('limits/time', 1200) # 20min time limit
m.readProblem(instance)


handler = EventHandler()
m.includeEventhdlr(handler, "handler" ,"")

m.optimize()
stats = handler.stats
m.writeStatistics("log.out")
m.freeProb()

with open('log.out', 'r+') as log:
    write = False
    file = open('testcsv.csv', 'w')
    for line in log:
        if line.split(':')[0] == "Diving Heuristic  ":
            write = True
            file.write(line)
            continue
        if line.split(':')[0] == "Neighborhoods      ":
            break
        if write:
            newLine = line.replace('|', ',')
            file.write(newLine)
    log.close()
    file.close()

with open('testcsv.csv', 'r+') as f:
    line = f.readline()

    line = f.readline()
    while len(line) > 1:
        heu = line.split(',')[0].strip()
        node = int(line.split(',')[1])
        # if( node not in stats.keys()):
        #     stats[node] = {}
        if('heuStats' not in stats[node].keys()):
            stats[node]['heuStats'] = {}
        stats[node]['heuStats'][heu] = []
        for i in range(8):
            stats[node]['heuStats'][heu].append(float(line.split(',')[i+2]))

        line = f.readline()
    f.close()




nodeIDInMatrix = []
nodeIDs = stats.keys()


# Node info in not saved by the order of node ID, so use a list to record the nodeIDs
nodeIDInMatrix = [i for i in nodeIDs]

successMatrix = np.zeros((len(nodeIDs), 17), dtype = int)
heuristics = ['actconsdiving', 'coefdiving', 'conflictdiving', 'crossover', 'distributiondiving', 'farkasdiving', 'fracdiving',
              'guideddiving', 'linesearchdiving', 'localbranching', 'pscostdiving', 'rens', 'rins', 'mutation', 'dins', 'trustregion',
              'veclendiving']



for i in range(len(nodeIDInMatrix)):

    nodeId = nodeIDInMatrix[i]
    print(stats[nodeId].keys())
    heusOfNode = stats[nodeId]['heuStats'].keys()
    for heu in range(len(heuristics)):
        if heuristics[heu] in heusOfNode:
            successMatrix[i][heu] = 1

print(stats[1]['heuStats'].keys())

Xtrain = []
for i in nodeIDs:
    Xtrain.append(stats[i]['features'])
Xtrain = np.array(Xtrain)

@ignore_warnings(category=ConvergenceWarning)
def training():
    for heuIndex in range(len(heuristics)):
        heuristic = heuristics[heuIndex]
        Ytrain = successMatrix[:, heuIndex].T.ravel()

        # Ytrain = np.array(Ytrain).reshape(len(Xtrain) ,1).ravel()

        lr = LogisticRegression()
        lr.fit(Xtrain, Ytrain)

        # to save the trained model
        fileName = heuristic + ".sav"
        joblib.dump(lr, fileName)

    #to load the model
    #loadedModel = joblib.load(fileName)

training()
