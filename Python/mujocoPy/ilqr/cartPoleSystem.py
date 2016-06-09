import os
import mujoco_py
from mujoco_py import mjviewer, mjcore
from mujoco_py import mjtypes
from mujoco_py import glfw
import numpy as np
import scipy.linalg as sp_linalg
import gym
from gym import error, spaces
from gym.utils import seeding
from os import path
import six
from ilqrClass import *

class cartPole():      
    xml_path = '/Users/michaelmathew/Documents/mjpro131/otherModels/cartoPole.xml'

    def __init__(self, solve_continuous=False): 
        if not path.exists(self.xml_path):
            raise IOError("File %s does not exist"%xml_path)
        self.model = mjcore.MjModel(self.xml_path)
        self.u = None
        self.dt = self.model.opt.timestep;
        self.start = None
        self.target = None
        numStates = 4
        numCtrls = 1

        self.Q = np.matrix(np.zeros((numStates,numStates)))
        self.Q[0,0] = 0; self.Q[1,1] = 100;
        self.R = np.matrix(np.eye(numCtrls))

        bounds = self.model.actuator_ctrlrange.copy()
        self.solve_continuous = False
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))} 
        
    def viewerSetup(self):
        self.width = 640
        self.height = 480
        self.viewer = mjviewer.MjViewer(visible=True,
                                        init_width=self.width,
                                        init_height=self.height)
        #self.viewer.cam.trackbodyid = 0 #2
        self.viewer.cam.distance = self.model.stat.extent * 0.0 #0.75
        self.viewer.cam.lookat[0] += 3 #0.8
        self.viewer.cam.elevation = 160
        self.viewer.start()
        self.viewer.set_model(self.model)
        #(data, width, height) = self.viewer.get_image()

    def viewerEnd(self):
        self.viewer.finish()
        self.viewer = None

    def viewerStart(self):
        if self.viewer is None:
            self.viewerSetup()
        return self.viewer       

    def viewerRender(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewerStart().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self.viewerStart().render()
            self.viewerStart().set_model(self.model)
            data, width, height = self.viewerStart().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        elif mode == 'human':
            self.viewerStart().loop_once()
                                               
    def resetModel(self):
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.resetModel()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewerSetup()
        return ob
        
    def getComPos(self):
        ridx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]
    
    def getComVel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]   
    
    def getXmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))
    
    def getStateVector(self, model):
        return np.concatenate([model.data.qpos.flat,
                               model.data.qvel.flat])
        
    def setState(self, model, q):
        dof = model.nv;
        qpos = q[0:dof]
        qvel = q[dof:dof*2]
        model.data.qpos = qpos
        model.data.qvel = qvel
        model._compute_subtree() 
        model.forward()
        model.data.ctrl = 0;
        model.step();
        return model
    
    def setControl(self, model, ctrl, nFrames):
        model.data.ctrl = ctrl
        for _ in range(nFrames):
            model.step()
        return model
    
        
    def linearizeModel(self, q, u):
        eps = 0.00001  # finite difference epsilon
        #----------- compute xdot_x and xdot_u using finite differences --------
        # NOTE: here each different run is in its own column
        dof = self.model.nv
        #eps = 1e-5
        #print q
        #print u
        A = np.zeros((dof*2, dof*2))
        for ii in range(dof*2):
            qtmp = np.copy(q)
            qtmp[ii] += eps
            q_inc = self.simulateFwd(qtmp, u);
            qtmp = np.copy(q)
            qtmp[ii] += -eps
            q_dec = self.simulateFwd(qtmp, u);
            A[:,ii] = (q_inc - q_dec) / (2 * eps)
        
        B = np.zeros((dof*2, len(u)))
        for ii in range(len(u)):
            utmp = np.copy(u);
            utmp[ii] += eps
            q_inc = self.simulateFwd(q, utmp);
            utmp = np.copy(u);
            utmp[ii] += -eps
            q_dec = self.simulateFwd(q, utmp);
            B[:,ii] = (q_inc - q_dec) / (2 * eps)

        return np.matrix(A), np.matrix(B)

    def simulateFwd(self, x, u): #simulate locally
        localModel = mjcore.MjModel(self.xml_path)
        dof = localModel.nv
        localModel = self.setState(localModel, x)
        localModel = self.setControl(localModel, u, 1)
        xnext = self.getStateVector(localModel)

        return xnext

    def computeCost(self, qArray, uArray, qDes):
        Q = self.Q
        R = self.R
        cost = 0.0
        tN = qList.shape[0]
        print(type(qVectList))
        print(typr(qVect[1]))
        for i in range(tN):
            qVect = qList[i]
            uVect = uList[i]
            cost += 0.5*(qVect-qDes).T*Q*(qVect-qDes) \
                  + 0.5*uVect.T*R*uVect
        cost += 1*(qArray[tN]-qDes).T*Q*(qArray[tN]-qDes)
        self.Q = Q
        self.R = R
        
        return cost
           
    def computeNextState(self, dt, qVect, uVect):
        return self.simulateFwd(qVect,uVect)
    
    def computeAllCostDeriv(self, qVect, uVect, qDes):
        Q = self.Q
        R = self.R
        numStates = qVect.shape[0]
        numCtrls = uVect.shape[0]
        lx = Q*(qVect-qDes)
        lxx = Q
        lu = R*uVect
        luu = R
        lux = np.matrix(np.zeros((numCtrls,numStates)))
        lxu = np.matrix(np.zeros((numStates,numCtrls)))

        return lx, lxx, lu, luu, lux, lxu

    def computeFinalCostDeriv(self, qtNVect, qDes):
        lx = 1.0*self.Q*(qtNVect-qDes)
        lxx = 1.0*self.Q

        return lx, lxx
    
    def computeAllModelDeriv(self, dt, qVect, uVect):
        fx,fu = self.linearizeModel(qVect, uVect)
        numStates = qVect.shape[0]
        numCtrls = uVect.shape[0]

        fxx = list(range(numStates))
        for i in range(numStates):
            fxx[i] = np.zeros((numStates,numStates))

        fuu = np.zeros((numStates,numCtrls))
        fux = np.zeros((numStates,numStates))
        fxu = np.zeros((numStates,numStates))

        return fx, fxx, fu, fuu, fxu, fux   


if __name__ == "__main__":

    myCartPole = cartPole()
    myCartPole.viewerSetup();
    crtPos = 0.0
    penPos = -90*3.14/180; # +/- 7degrees
    crtVel= 0.0; #initial allowed deviation Position +/- 0.31
    penVel = 0*3.14/180; #initial allowed deviation Velocity
    qinit = np.matrix([[crtPos],[penPos],[crtVel],[penVel]])
    myCartPole.setState(myCartPole.model,qinit)
    
    myCartPole.numStates = 4
    myCartPole.numCtrls = 1
    sysModel = cartPole()
    sysModel.start = qinit
    sysModel.target = np.matrix([[0.0],[0.0],[0.0],[0.0]])
    sysModel.numCtrls = 1
    sysModel.numStates = 4

    ilqrSoln = ilqrSolver(sysModel)
    qTraj, uTraj,_ = ilqrSoln.findTrajectory(sysModel.start,sysModel.target,sysModel.dt,20)
    print(uTraj)
   
    myCartPole.viewer = myCartPole.viewerStart();
    for i in range(len(uTraj)):
        myCartPole.viewerRender()
        myCartPole.setControl(myCartPole.model, uTraj[i], 1)
    myCartPole.viewerEnd()
        


    