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

        return A, B

    def simulateFwd(self, x, u): #simulate locally
        localModel = mjcore.MjModel(self.xml_path)
        dof = localModel.nv
        localModel = self.setState(localModel, x)
        localModel = self.setControl(localModel, u, 1)
        xnext = self.getStateVector(localModel)

        return xnext

    def lqr(self, A, B, Q, R):
        P = np.matrix(sp_linalg.solve_continuous_are(A, B, Q, R)) #solve continous time ricatti equation
        K = np.matrix(sp_linalg.inv(R)*(B.T*P)) #compute the LQR gain
        eigVals, eigVecs = sp_linalg.eig(A-B*K)

        return K, eigVals
 
    def dlqr(self, A, B, Q, R):
        P = np.matrix(sp_linalg.solve_discrete_are(A, B, Q, R))#solve discrete time ricatti equation
        K = np.matrix(sp_linalg.inv(B.T*P*B+R)*(B.T*P*A))#compute the LQR gain
        eigVals, eigVecs = sp_linalg.eig(A-B*K)
        
        return K, eigVals
        
    def lqrControl(self, x_des=None):

        dof = self.model.nv
        if self.u is None:
            self.u = np.zeros(dof-1)

        C = np.matrix([[1,0,0,0],[0,1,0,0]]);
        #self.Q = np.zeros((dof*2, dof*2))
        Q = np.dot(C.T,C);
        Q[:dof, :dof] = np.eye(dof) * 1000.0 #final position weighting matrix
        R = np.eye(dof-1) * self.dt #final control weighting matrix
        #self.R[0,0] = 1;
        #print(self.R)
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
            K, eigVals = self.lqr(A, B, Q, R)
        else:
            K, eigVals = self.dlqr(A, B, Q, R)
             
        '''print('A Matrix')
        print(A)
        print('B Matrix')
        print(B)
        print('K Matrix')
        print(K)
        print('A-BK Matrix')
        print(A-B*K)
        print('Open loop poles')
        print(sp_linalg.eigvals(A).T)
        print('Closed loop poles')
        print(eigVals)'''

        # if the target is given in cartesian space , it should be transfered to
        #joint space using jacobian.
        #J = self.getJacobian() -> not implemented
        #u = np.hstack([np.dot(J.T, self.target), self.model.data.qvel])
        #self.u = -np.dot(K,u)
        #print(K)
        self.u = -np.dot(K,self.getStateVector(self.model))
        #print(self.u)
        return self.u

    '''def softmax(w, t = 1.0):
        e = np.exp(npa(w) / t)
        dist = e / np.sum(e)
    return dist'''
    
    def swingUp(self, q_curr):
        jIM = self.model.data.qM #joint space inertia matrix
        g = 9.81
        energy0 = 0#jIM[0]*g*0.5*(1-np.cos(np.pi))
        energy = 0.5*jIM[0]*q_curr[3]**2 + jIM[0]*9.81*0.5*(1-np.cos(q_curr[1]));
        gain = 6*1e-1;
        n = 3.0/9.81;
        xlim = np.sin(np.pi/2*q_curr[0])
        #print(xlim)
        u = xlim*n*g*np.sign(gain*(energy-energy0)*q_curr[3]*np.cos(q_curr[1]))
        u0 = 0.0;
        k = 1.5;
        self.u = -3.0 + 3.0/(1+np.exp(-k*(u-u0))); #logistic function

        if(((q_curr[0] > 0) or (q_curr[0] < 1)) and (q_curr[2] < 0)):
            pass
        elif((q_curr[0] < 0) or (q_curr[0] > -1) and (q_curr[2] > 0)):
            pass
        else:
            self.u = -self.u
                  
        return self.u
    
    def switchingController(self):
        q_curr = self.getStateVector(self.model)
        if(np.absolute(q_curr[1]) < 0.349): #20 degrees
            u = self.lqrControl()
            print("dlqr turned on")
        else:
            u = self.swingUp(q_curr)
            print("swing up turned on")
        
        return u

if __name__ == "__main__":
    myCartPole = cartPoleLQR()
    myCartPole.viewerSetup();
    crtPos = 0
    penPos = -17*3.14/180; # +/- 7degrees
    crtVel= 0.0; #initial allowed deviation Position +/- 0.31
    penVel = 0*3.14/180; #initial allowed deviation Velocity
    myCartPole.setState(myCartPole.model,[crtPos,penPos,crtVel,penVel])
    #myCartPole.viewer = myCartPole.viewerStart();
    #myCartPole.swingUp()
    for ii in range(500):
        myCartPole.viewerRender();
        #print(myCartPole.u)
        #u = myCartPole.lqrControl();
        u = myCartPole.switchingController();
        myCartPole.setControl(myCartPole.model, u, 1)
    #

    
        
    

    

    

   