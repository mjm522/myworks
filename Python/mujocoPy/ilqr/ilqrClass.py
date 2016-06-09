import numpy as np
import numpy.linalg
import matplotlib.pyplot as pl
import time

class ilqrSolver:

    def __init__(self, sysModel):
        self.sysModel = sysModel
        self.costFn = sysModel.computeCost
        ''' init values changed when solveTrajectory is called '''
        self.Xinit = sysModel.start
        self.Xdes = sysModel.target
        self.tN = 10
        self.dt = 1e-4
        self.maxIter = 20
        self.epsConv = 1e-3
        self.changeAmnt = 0.0

    def findTrajectory(self,Xinit,Xdes,dt,tN =10, maxIter=20, epsConv=1e-3):
        self.Xinit = Xinit
        self.Xdes = Xdes
        self.tN = tN
        self.dt = dt
        self.matIter = maxIter
        self.epsConv = epsConv

        Xtraj, Utraj = self.initTraj()
        for itr in range(maxIter):
            ktraj, Ktraj = self.bkwdPass(Xtraj,Utraj)
            newXtraj, newUtraj = self.fwdPass(ktraj,Ktraj,Xtraj,Utraj)
            Xtraj = newXtraj
            Utraj = newUtraj

            if(self.changeAmnt < self.epsConv):
                break
        return Xtraj,Utraj, itr

    def initTraj(self):
        Xtraj = []
        dt = self.dt
        numCtrls = self.sysModel.numCtrls
        Utraj = [np.matrix(np.zeros((numCtrls,1))) for i in range(self.tN)]
        Xtraj.append(self.Xinit)
        computeNextState = self.sysModel.computeNextState
        for i in range(self.tN):
            Xnew = np.matrix(computeNextState(dt,Xtraj[i], Utraj[i]))
            Xtraj.append(Xnew.T)
        return Xtraj, Utraj
    
    def bkwdPass(self,Xtraj,Utraj):
        ktraj = []
        Ktraj = []
        Xdes = self.Xdes
        numStates = self.sysModel.numStates
        dt = self.dt

        nxtVx, nxtVxx = self.sysModel.computeFinalCostDeriv(Xtraj[self.tN], self.Xdes)
        cmptModelDeriv = self.sysModel.computeAllModelDeriv
        cmptCstDeriv = self.sysModel.computeAllCostDeriv

        mu = 0.0
        cmpltBkwdPassFlag = 0
        while(cmpltBkwdPassFlag == 0):
            cmpltBkwdPassFlag = 1
            mueye = mu*np.eye(nxtVxx.shape[0], dtype=float)
            for i in range(self.tN-1,-1,-1):
                X = Xtraj[i]
                U = Utraj[i]

                fx,fxx,fu,fuu,fxu,fux = cmptModelDeriv(dt,X,U)
                lx, lxx, lu, luu, lux, lxu =  cmptCstDeriv(X,U,Xdes)

                Qx   = lx  + fx.T*nxtVx
                Qu   = lu  + fu.T*nxtVx
                Qxx  = lxx + fx.T*nxtVxx*fx
                Quut = luu + fu.T*(nxtVxx+mueye)* fu
                Quxt = lux + fu.T*(nxtVxx+mueye)*fx
                
                for j in range(numStates):
                    Qxx  += nxtVx[j].item()*fxx[j]
                    Quxt += nxtVx[j].item()*fux[j]
                    Quut += nxtVx[j].item()*fuu[j]

                if(np.any(np.linalg.eigvals(Quut) <= 1e-10)):
                    if(mu == 0):
                        mu += 1e-4
                    else:
                        mu = mu*10
                    cmpltBkwdPassFlag = 0
                    break
                
                QuutInv = -np.linalg.inv(Quut)
                k = QuutInv*Qu
                K = QuutInv*Quxt

                nxtVx  = Qx  - K.T*Quut*k
                nxtVxx = Qxx - K.T*Quut*K

                ktraj.append(k)
                Ktraj.append(K)
        ktraj.reverse()
        Ktraj.reverse()
        return ktraj,Ktraj

    def fwdPass(self,ktraj,Ktraj,Xtraj,Utraj):
        newXtraj = []
        newUtraj = []
        changeAmnt = 0.0
        dt = self.dt
        numCtrls = self.sysModel.numCtrls
        newXtraj.append(self.Xinit)
        alphalist = [1.0,0.8,0.6,0.4,0.2] #line search to be implemented
        alpha = alphalist[0]
        computeNextState = self.sysModel.computeNextState
        for i in range(self.tN):
            newUtraj.append(Utraj[i] + alpha*ktraj[i] + Ktraj[i]*(newXtraj[i]-Xtraj[i]))
            Xnew = np.matrix(computeNextState(dt,Xtraj[i], Utraj[i]))
            newXtraj.append(Xnew.T)
            for j in range(numCtrls):
                U = Utraj[i]
                newU = newUtraj[i]
                changeAmnt += np.abs(U[j] - newU[j])
        self.changeAmnt = changeAmnt
        return newXtraj, newUtraj