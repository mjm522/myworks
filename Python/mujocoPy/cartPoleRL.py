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

class cartPoleLQR():      
    xml_path = '/Users/michaelmathew/Documents/VisualStudioCode/Mujoco/Control/matlab/cartoPole/cartoPole.xml'

    def __init__(self, solve_continuous=False): 
        if not path.exists(self.xml_path):
            raise IOError("File %s does not exist"%xml_path)
        self.model = mjcore.MjModel(self.xml_path)
        self.u = None
        self.dt = self.model.opt.timestep;
        self.target = [0,0,0,0];
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
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -60
        self.viewer.start()
        self.viewer.set_model(self.model)

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
        #print(q)
        qpos = q[0:dof]
        #print(qpos)
        qvel = q[dof:dof*2]
        #print(qvel)
        #assert qpos.shape == (model.nq,) and qvel.shape == (model.nv,)
        model.data.qpos = qpos
        model.data.qvel = qvel
        model._compute_subtree() #pylint: disable=W0212
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
        A = np.zeros((dof*2, dof*2))
        for ii in range(dof*2):
            qtmp = q
            qtmp[ii] += eps
            q_inc = self.simulateFwd(qtmp, u);
            qtmp = q
            qtmp[ii] -= eps
            q_dec = self.simulateFwd(qtmp, u);
            A[:,ii] = (q_inc - q_dec) / (2 * eps)
        
        B = np.zeros((dof*2, 1))
        for ii in range(1):
            utmp = u;
            utmp[ii] += eps
            q_inc = self.simulateFwd(q, utmp);
            utmp = u;
            utmp[ii] -= eps
            q_dec = self.simulateFwd(q, utmp);
            B[:,ii] = (q_inc - q_dec) / (2 * eps)

        return A, B

    def simulateFwd(self, x, u):
        """ Simulate the arm dynamics locally. """
        localModel = mjcore.MjModel(self.xml_path)
        dof = localModel.nv
        localModel = self.setState(localModel, x)
        localModel = self.setControl(localModel, u, 1)
        xnext = self.getStateVector(localModel)
        
        return xnext
        
    def lqrControl(self, x_des=None):

        dof = self.model.nv
        if self.u is None:
            self.u = np.zeros(dof-1)

        self.C = np.matrix([[1,0,0,0],[0,1,0,0]]);
        #self.Q = np.zeros((dof*2, dof*2))
        self.Q = np.dot(self.C.T,self.C);
        self.Q[:dof, :dof] = np.eye(dof) * 1000.0 #final position weighting matrix
        self.R = np.eye(dof-1) * self.dt #final control weighting matrix
        #self.R[0,0] = 1;
        print(self.R)
        # calculate desired end-effector acceleration
        if x_des is None:
            x_des = self.getStateVector(self.model) - self.target 

        #self.arm, state = self.getStateVector(self.model)
        state = self.getStateVector(self.model)
        '''print('state')
        print(state)
        print('control')
        print(self.model.data.ctrl)'''
        A, B = self.linearizeModel(state, self.u)

        if self.solve_continuous is True:
            P = sp_linalg.solve_continuous_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R), np.dot(B.T, P))
        else:
            #print(A)
            #print(self.Q) 
            P = sp_linalg.solve_discrete_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))
            print('A Matrix')
            print(A)
            print('B Matrix')
            print(B)
            print('K Matrix')
            print(K)
            print('A-BK Matrix')
            print(A-np.dot(B,K))
            print('Open loop poles')
            print(sp_linalg.eigvals(A).T)
            print('Closed loop poles')
            print(sp_linalg.eigvals(A-np.dot(B,K)).T)

        # if the target is given in cartesian space , it should be transfered to
        #joint space using jacobian.
        #J = self.getJacobian() -> not implemented
        #u = np.hstack([np.dot(J.T, self.target), self.model.data.qvel])
        #self.u = -np.dot(K,u)
        #print(K)
        self.u = -np.dot(K, self.getStateVector(self.model))
        return self.u

if __name__ == "__main__":
    myCartPole = cartPoleLQR()
    myCartPole.viewerSetup();
    #myCartPole.viewer = myCartPole.viewerStart();
    for ii in range(1):
        myCartPole.viewerRender();
        #print(myCartPole.u)
        u = myCartPole.lqrControl();
        myCartPole.setControl(myCartPole.model, u, 1)
    #

    
        
    

    

    

   