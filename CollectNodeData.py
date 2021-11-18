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
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
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
                if nodeID >1:
                    cost = var.getObj()
                    cols = var.getCol()
                    temp = np.array(cols.getVals())
                    LPSol = var.getLPSol()
                    frac = LPSol - np.floor(LPSol)
                    if (cost > 0):
                        vec_length = ( 1 -frac) * cost / (np.linalg.norm(temp ) +1)
                    else:
                        vec_length = - frac * cost / (np.linalg.norm(temp) + 1)
                    vectorLen. append(vec_length)
                else:
                    vectorLen.append(0)
                pseudocost.append(self.model.getVarPseudocost(var))

            # Sum of variables’ LP solution fractionalities / Num. of Fractional Variables
            if nOfFrac == 0:
                nodeFeatures.append(1)
            else:
                nodeFeatures.append(np.sum(fractionalities) / nOfFrac)

            # Num. of Fractional Variable / Num. of Integer Variables
            if nOfFrac == 0:
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
                nodeFeatures.append(np.mean(vec_length))
                nodeFeatures.append(np.min(vec_length))
                nodeFeatures.append(np.max(vec_length))
                nodeFeatures.append(np.std(vec_length))
                nodeFeatures.append(np.median(vec_length))

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
    for line in log:
        if line == "Diving (single)    :      Calls      Nodes   LP Iters Backtracks  Conflicts   MinDepth   MaxDepth   AvgDepth  RoundSols  NLeafSols  MinSolDpt  MaxSolDpt  AvgSolDpt":
            write = True
            continue
        if line == "Neighborhoods      :      Calls  SetupTime  SolveTime SolveNodes       Sols       Best       Exp3  EpsGreedy        UCB TgtFixRate  Opt  Inf Node Stal  Sol  Usr Othr Actv":
            break
        if write:
            file = open('NodeStats.csv', 'w')
            file.write(line)

with open('NodeStats.csv', 'r+') as f:
    # read file
    file_source = f.read()
    replace_string = file_source.replace('|', ',')
    # save output
    f.write(replace_string)

    f.close()


with open('Nodestats.csv', 'r+') as f:
    line = f.readline()

    line = f.readline()
    while line:
        heu = line.split(',')[0]
        node = int(line.split(',')[1])
        stats[node]['heuStats'] = {}
        stats[node]['heuStats'][heu] = []
        for i in range(8):
            stats[node]['heuStats'][heu].append(line.split(',')[i+2])

        line = f.readline()
    f.close()

