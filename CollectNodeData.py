from pyscipopt import scip
import pyscipopt as pyscipopt
import numpy as np


class EventHandler(pyscipopt.Eventhdlr):
    """
    A SCIP event handler that records solving stats
    """
    def __init__(self):
        self.stats = {}

    def eventinit(self):
        #self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        #self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexit(self):
        #self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        #self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexec(self, event):

        event_type = event.getType()

        if(event_type == pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED):

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

            vars = self.model.getVars()
            fracVars = []
            nOfFrac = 0
            fractionalities = []
            for var in vars:
                LPSol = var.getLPSol()
                if abs(LPSol- int(LPSol)) < 0.00001:
                    fracVars.append(var)
                    nOfFrac += 1
                    fractionalities.append(LPSol - np.floor(LPSol))

            # Error occurs:LPCan = self.model.getLPBranchCands()
            # fracVars = LPCan[0]
            # fractionalities = LPCan[2]
            # nOfFrac = LPCan[3]
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
                vectorLen. append(vec_length)

                pseudocost.append(self.model.getVarPseudocost(var))

            # Sum of variables’ LP solution fractionalities / Num. of Fractional Variables
            if nOfFrac == 0:
                nodeFeatures.append(1)
            else:
                nodeFeatures.append(np.sum(fractionalities) / nOfFrac)

            # Num. of Fractional Variable / Num. of Integer Variables
            if (len(vars) - nOfFrac) == 0:
                nodeFeatures.append(1)
            else:
                nodeFeatures.append(nOfFrac / (len(vars) - nOfFrac))

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
            print(self.stats[nodeID]['features'])



instance = "ins.lp"


m = scip.Model()
m.readParams("params.set")
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
        if( node not in stats.keys()):

            stats[node] = {}
            stats[node]['heuStats'] = {}
        stats[node]['heuStats'][heu] = []
        for i in range(8):
            stats[node]['heuStats'][heu].append(float(line.split(',')[i+2]))

        line = f.readline()
    f.close()


