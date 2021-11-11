
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
            nodeStats = []


            # optimality gap
            optimality_gap = self.model.getGap()
            nodeStats.append(optimality_gap)

            # infinity gap
            isInf = self.model.isInfinity(optimality_gap)
            nodeStats.append(isInf)

            # Root LP value / Global Lower Bound
            nodeStats.append(self.model.getDualboundRoot( ) /self.model.getLPObjVal())

            # Root LP value / Global Upper Bound
            nodeStats.append(self.model.getDualboundRoot( ) /(self.model.getLPObjVal( ) +optimality_gap))

            # Node Depth
            nodeStats.append(self.model.getDepth())

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
                nodeStats.append(1)
            else:
                nodeStats.append(np.sum(fractionalities) / nOfFrac)

            # Num. of Fractional Variable / Num. of Integer Variables
            if nOfFrac == 0:
                nodeStats.append(1)
            else:
                nodeStats.append(nOfFrac / (len(vars) - nOfFrac))

            # Num. of Active Constraints / Num. of Constraints
            conss = self.model.getConss()
            nOfActiveCons = 0
            for cons in conss:
                if(cons.isActive()):
                    nOfActiveCons += 1
            nodeStats.append(nOfActiveCons/ len(conss))

            # Node is root
            if(nodeID == 1):
                nodeStats.append(True)
            else:
                nodeStats.append(False)

            # Root LP value / Node LB
            nodeStats.append(self.model.getDualboundRoot() / self.model.getCurrentNode().getLowerbound())
            # Root LP value / Node Estimate

            # Node estimate
            estimate = self.model.getCurrentNode().getEstimate()
            nodeStats.append(self.model.getDualboundRoot() / estimate)

            # Number of Up Locks
            if nOfFrac == 0:
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
            else:
                nodeStats.append(np.mean(nOfUpLocks))
                nodeStats.append(np.min(nOfUpLocks))
                nodeStats.append(np.max(nOfUpLocks))
                nodeStats.append(np.std(nOfUpLocks))
                nodeStats.append(np.median(nOfUpLocks))

            # Number of Down Locks
            if nOfFrac == 0:
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
            else:
                nodeStats.append(np.mean(nOfDownLocks))
                nodeStats.append(np.min(nOfDownLocks))
                nodeStats.append(np.max(nOfDownLocks))
                nodeStats.append(np.std(nOfDownLocks))
                nodeStats.append(np.median(nOfDownLocks))

            # Normalized Objective Coefﬁcient

            # Distance to root LP solution
            if nOfFrac == 0:
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
            else:
                nodeStats.append(np.mean(distanceToLPSol))
                nodeStats.append(np.min(distanceToLPSol))
                nodeStats.append(np.max(distanceToLPSol))
                nodeStats.append(np.std(distanceToLPSol))
                nodeStats.append(np.median(distanceToLPSol))

            # Vector Length
            if nOfFrac == 0:
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
            else:
                nodeStats.append(np.mean(vec_length))
                nodeStats.append(np.min(vec_length))
                nodeStats.append(np.max(vec_length))
                nodeStats.append(np.std(vec_length))
                nodeStats.append(np.median(vec_length))

            # Pseudocost score
            if nOfFrac == 0:
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
                nodeStats.append(0)
            else:
                nodeStats.append(np.mean(pseudocost))
                nodeStats.append(np.min(pseudocost))
                nodeStats.append(np.max(pseudocost))
                nodeStats.append(np.std(pseudocost))
                nodeStats.append(np.median(pseudocost))

            self.stats[nodeID] = nodeStats



instance = "ins.lp"


m = scip.Model()
m.readParams("params.set")
m.readProblem(instance)


handler = EventHandler()
m.includeEventhdlr(handler, "handler" ,"")

m.optimize()
m.writeStatistics("log.out")
m.freeProb()

