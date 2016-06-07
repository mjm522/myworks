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
        self.start = None
        self.target = [0.0,0.0,0.0,0.0];
        bounds = self.model.actuator_ctrlrange.copy()
        self.solve_continuous = False
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))}
        #ILQR parameters
        self.t = 0
        self.tN = 50 #number of time steps
        self.maxIter = 100 #maximum iterations
        self.lambFctr = 10
        self.lambMx = 1000
        self.epsConv = 1e-3 #converge threshold
        self.U = np.zeros((self.tN,1))
        
    
        
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
        
    def forwardSimulate(self, q0, U):
        """ do a rollout of the system, starting at x0 and 
        applying the control sequence U
        x0 np.array: the initial state of the system
        U np.array: the control sequence to apply
        """ 
        tN = U.shape[0]
        numStates = q0.shape[0]
        dt = self.dt

        X = np.zeros((tN, numStates))
        X[0] = q0
        cost = 0

        # Run simulation with substeps
        for t in range(tN-1):
            X[t+1] = self.simulateFwd(X[t], U[t])
            l,_,_,_,_,_ = self.costImm(X[t], U[t])
            cost = cost + dt * l

        # Adjust for final cost, subsample trajectory
        lf,_,_ = self.costFnl(X[-1])
        cost = cost + lf
        print(q0)
        return X, cost
        
    def ilqr(self, x0, U=None): 
        """ use iterative linear quadratic regulation to find a control 
        sequence that minimizes the cost function 
        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        U = self.U if U is None else U

        tN = U.shape[0] # number of time steps
        dof = self.model.nv # number of degrees of freedom of plant 
        numStates = dof * 2 # number of states (position and velocity)
        numCtrls = U.shape[1];
        dt = self.dt # time step

        lamb = 1.0 # regularization parameter
        sim_new_trajectory = True

        for ii in range(self.maxIter):

            if sim_new_trajectory == True: 
                # simulate forward using the current control trajectory
                X, cost = self.forwardSimulate(x0, U)
                oldcost = np.copy(cost) # copy for exit condition check
                #print(X)
                # now we linearly approximate the dynamics, and quadratically 
                # approximate the cost function so we can use LQR methods 

                # for storing linearized dynamics
                # x(t+1) = f(x(t), u(t))
                fx = np.zeros((tN, numStates, numStates)) # df / dx
                fu = np.zeros((tN, numStates, numCtrls)) # df / du
                # for storing quadratized cost function 
                l = np.zeros((tN,1)) # immediate state cost 
                lx = np.zeros((tN, numStates)) # dl / dx
                lxx = np.zeros((tN, numStates, numStates)) # d^2 l / dx^2
                lu = np.zeros((tN, numCtrls)) # dl / du
                luu = np.zeros((tN, numCtrls, numCtrls)) # d^2 l / du^2
                lux = np.zeros((tN, numCtrls, numStates)) # d^2 l / du / dx
                # for everything except final state
                for t in range(tN-1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    #print(X[t]); print(U[t])
                    A, B = self.linearizeModel(X[t], [U[t]])
                    fx[t] = np.eye(numStates) + A * dt
                    fu[t] = B * dt
                
                    (l[t], lx[t], lxx[t], lu[t], 
                        luu[t], lux[t]) = self.costImm(X[t], U[t])
                    l[t] *= dt
                    lx[t] *= dt
                    lxx[t] *= dt
                    lu[t] *= dt
                    luu[t] *= dt
                    lux[t] *= dt
                # aaaand for final state
                l[-1], lx[-1], lxx[-1] = self.costFnl(X[-1])

                sim_new_trajectory = False

            # optimize things! 
            # initialize Vs with final state cost and set up k, K 
            V = l[-1].copy() # value function
            Vx = lx[-1].copy() # dV / dx
            Vxx = lxx[-1].copy() # d^2 V / dx^2
            k = np.zeros((tN, numCtrls)) # feedforward modification
            K = np.zeros((tN, numCtrls, numStates)) # feedback gain

            # NOTE: they use V' to denote the value at the next timestep, 
            # they have this redundant in their notation making it a 
            # function of f(x + dx, u + du) and using the ', but it makes for 
            # convenient shorthand when you drop function dependencies

            # work backwards to solve for V, Q, k, and K
            for t in range(tN-2, -1, -1):

                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Qx = lx[t] + np.dot(fx[t].T, Vx) 
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                Qu = lu[t] + np.dot(fu[t].T, Vx)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                
                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Qxx = lxx[t] + np.dot(fx[t].T, np.dot(Vxx, fx[t])) 
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Qux = lux[t] + np.dot(fu[t].T, np.dot(Vxx, fx[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                Quu = luu[t] + np.dot(fu[t].T, np.dot(Vxx, fu[t]))

                # Calculate Q_uu^-1 with regularization term set by 
                # Levenberg-Marquardt heuristic (at end of this loop)
                Quu_evals, Quu_evecs = np.linalg.eig(Quu)
                Quu_evals[Quu_evals < 0] = 0.0
                Quu_evals += lamb
                Quu_inv = np.dot(Quu_evecs, 
                        np.dot(np.diag(1.0/Quu_evals), Quu_evecs.T))

                # 5b) k = -np.dot(Q_uu^-1, Q_u)
                k[t] = -np.dot(Quu_inv, Qu)
                # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                K[t] = -np.dot(Quu_inv, Qux)

                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                Vx = Qx - np.dot(K[t].T, np.dot(Quu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                Vxx = Qxx - np.dot(K[t].T, np.dot(Quu, K[t]))

            Unew = np.zeros((tN, numCtrls))
            # calculate the optimal change to the control trajectory
            xnew = x0.copy() # 7a)
            for t in range(tN - 1): 
                # use feedforward (k) and feedback (K) gain matrices 
                # calculated from our value function approximation
                # to take a stab at the optimal control signal
                Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t]) # 7b)
                # given this u, find our next state
                xnew = self.simulateFwd(xnew, Unew[t]) # 7c)

            # evaluate the new trajectory 
            Xnew, costnew = self.forwardSimulate(x0, Unew)

            # Levenberg-Marquardt heuristic
            if costnew < cost: 
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lambFctr

                X = np.copy(Xnew) # update trajectory 
                U = np.copy(Unew) # update control signal
                oldcost = np.copy(cost)
                cost = np.copy(costnew)

                sim_new_trajectory = True # do another rollout

                # print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldcost-cost)/cost) < self.epsConv):
                    print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                            " logLambda = %.1f"%np.log(lamb))
                    break

            else: 
                # increase lambda (get closer to gradient descent)
                lamb *= self.lambFctr
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lambMx: 
                    print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                          np.log(lamb)))
                    break

        #print('sizes x')
        #print(len(X)); 
        #print('size u')
        #print(len(U)); #print(len(cost))
        print(X)
        return X, U, cost
        
    def ilqrControl(self, x_des=None):
        dof = self.model.nv
        if(self.t >= self.tN-1): #reset the time
            self.t = 0
        if self.t %1 == 0:
            #print(self.start)
            x0 = self.start
            U = np.copy(self.U[self.t:])
            self.X, self.U[self.t:], cost = self.ilqr(x0,U)
        self.u = self.U[self.t]
        
        self.t += 1 #move a step forward in the ctrl sequence
        
        return self.u
        
    def costImm(self,q,u): #immediate state cost function
        dof = self.model.nv
        numStates = q.shape[0]
        numCtrls  = u.shape[0]
        l = np.sum(u**2)
        lx = np.zeros(numStates)
        lxx = np.zeros((numStates,numStates))
        lu = 2*u
        luu = 2*np.eye(numCtrls)
        lux = np.zeros((numCtrls,numStates))
        
        return l, lx, lxx, lu, luu, lux
        
    def costFnl(self,q):
        numStates = q.shape[0]
        dof = self.model.nv
        
        lx = np.zeros(numStates)
        lxx = np.zeros((numStates,numStates))
        
        wp = 1e4 #terminal position cost weight
        wv = 1e4 #terminal velocity cost weight
        
        q1q2 = self.getStateVector(self.model)
        q1q2_err = np.array([q1q2[0]-self.target[0], q1q2[1]-self.target[1]])
        l = (wp*np.sum(q1q2_err**2)+ wv*np.sum(q[dof:dof*2]**2))
        lx[0:dof] = 2*wp*q[0:dof]
        lx[dof:dof*2] = 2*wv*q[dof:dof*2]
        
        eps = 1e-4
        for k in range(dof): 
            veps = np.zeros(dof)
            veps[k] = eps
            q1 = self.simulateFwd(q[k]+veps,[0.0])
            lx1 = 2*wp*q1[0:dof]
            q2 = self.simulateFwd(q[k]-veps,[0.0])
            lx2 = 2*wp*q2[0:dof]
            lxx[0:dof, k] = ((lx1-lx2) / 2.0 / eps).flatten()
            #d1 = wp * self.difEnd(x[0:dof] + veps)
            #d2 = wp * self.difEnd(x[0:dof] - veps)
            #lxx[0:dof, k] = ((d1-d2) / 2.0 / eps).flatten()
        
        lxx[dof:dof*2, dof:dof*2] = 2 * wv * np.eye(dof)
        
        return l, lx, lxx

if __name__ == "__main__":
    myCartPole = cartPoleLQR()
    #myCartPole.viewerSetup();
    crtPos = 0
    penPos = -7*3.14/180; # +/- 7degrees
    crtVel= 0.0; #initial allowed deviation Position +/- 0.31
    penVel = 0*3.14/180; #initial allowed deviation Velocity
    myCartPole.start = np.array([crtPos,penPos,crtVel,penVel])
    #myCartPole.setState(myCartPole.model,[0.0,devPos,0.0,devVel])
    #myCartPole.viewerRender();
    #print(myCartPole.u)
    u = myCartPole.ilqrControl();
    print(u)
    #myCartPole.setControl(myCartPole.model, u, 1)   