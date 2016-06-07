from casadi import *
import numpy as NP
import matplotlib.pyplot as plt


class vdpMultiShoot():
    def __init__(self):
        self.x0=SX.sym("x0") # Declare variables (use simple, efficient DAG)
        self.x1=SX.sym("x1")
        self.x = vertcat(self.x0,self.x1)
        self.u = SX.sym("u") # Control
        self.nX = 4 # Augmented DAE state dimension
        self.tf = 10.0 # End time   
        self.num_nodes = 20 # Number of shooting nodes
        self.tgrid = NP.linspace(0,self.tf,100) # Time grid for visualization
        self.cartPoleODE()
        self.lagrangian()
        self.costate()
        self.hamiltonian()
        self.costateEqn()
     
    def cartPoleODE(self): # ODE right hand side
        self.xdot = vertcat((1 - self.x1*self.x1)*self.x0 - self.x1 + self.u, self.x0)

    def lagrangian(self): # Lagrangian function
        self.L = self.x0*self.x0 + self.x1*self.x1 + self.u*self.u
    
    def costate(self):
        self.lam = SX.sym("lam",2) # Costate

    def hamiltonian(self):
        self.H = dot(self.lam,self.xdot) + self.L  # Hamiltonian function
        print "Hamiltonian: ",self.H ## The control must minimize the Hamiltonian, which is:
        
    def costateEqn(self):
        self.ldot = -gradient(self.H,self.x) # Costate equations
    
    def integratorNLP(self, dae):
        iopts = {} # Create an integrator (CVodes)
        iopts["abstol"] = 1e-8 # abs. tolerance
        iopts["reltol"] = 1e-8 # rel. tolerance
        iopts["t0"] = 0.0
        iopts["tf"] = self.tf/self.num_nodes
        return integrator("I", "cvodes", dae, iopts)     
    
    def constraints(self):
        # H is of a convex quadratic form in u: H = u*u + p*u + q, let's get the coefficient p
        p = gradient(self.H,self.u)     # this gives us 2*u + p
        p = substitute(p,self.u,0) # replace u with zero: gives us p
        # H's unconstrained minimizer is: u = -p/2
        u_opt = -p/2
        # We must constrain u to the interval [-0.75, 1.0], convexity of H ensures that the optimum is obtain at the bound when u_opt is outside the interval
        u_opt = fmin(u_opt,1.0)
        u_opt = fmax(u_opt,-0.75)
        print "optimal control: ", u_opt
        f = vertcat(self.xdot,self.ldot) # Augment f with lam_dot and subtitute in the value for the optimal control
        f = substitute(f,self.u,u_opt)
        self.u_fcn = Function("ufcn", [vertcat(self.x,self.lam)], [u_opt]) # Function for obtaining the optimal control from the augmented state
        self.dae = {'x':vertcat(self.x,self.lam), 'ode':f} # Formulate the DAE
     
    def setupNLP(self):
        NV = self.nX*(self.num_nodes+1) # Variables in the root finding problem
        V = MX.sym("V",NV)    
        X = [] # Get the state at each shooting node
        v_offset = 0
        for k in range(self.num_nodes+1):
            X.append(V[v_offset:v_offset+self.nX])
            v_offset = v_offset+self.nX       
        G = [] # Formulate the root finding problem
        G.append(X[0][:2] - NP.array([0,1])) # states fixed, costates free at initial time
        I = self.integratorNLP(self.dae)
        for k in range(self.num_nodes):
            XF = I(x0=X[k])["xf"]
            G.append(XF-X[k+1])
        G.append(X[self.num_nodes][2:] - NP.array([0,0])) # costates fixed, states free at final time
        self.rfp = Function('rfp', [V], [vertcat(*G)]) # Terminal constraints: lam = 0
        
    def solveNLP(self): 
        opts = {} # Solver options
        opts["nlpsol"] = "ipopt"
        opts["nlpsol_options"] = {"ipopt.hessian_approximation":"limited-memory"}
        solver = rootfinder('solver', "nlpsol", self.rfp, opts) # Allocate a solver
        self.V_sol = solver(0) # Solve the problem
        
    def simulateResults(self):
        simulator = integrator('simulator', 'cvodes', self.dae, {'grid':self.tgrid,'output_t0':True}) # Simulator to get optimal state and control trajectories
        self.sol = simulator(x0 = self.V_sol[0:4])["xf"] # Simulate to get the trajectories
        ufcn_all = self.u_fcn.map("ufcn_all", "serial", len(self.tgrid)) # Calculate the optimal control
        self.u_opt = ufcn_all(self.sol)
        
    def plotSoln(self): # Plot the results
        plt.figure(1)
        plt.clf()
        plt.plot(self.tgrid, self.sol[0, :].T, '--')
        plt.plot(self.tgrid, self.sol[1, :].T, '-')
        plt.plot(self.tgrid, self.u_opt.T, '-.')
        plt.title("Van der Pol optimization - indirect multiple shooting")
        plt.xlabel('time')
        plt.legend(['x trajectory','y trajectory','u trajectory'])
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    vdp = vdpMultiShoot()
    vdp.constraints()
    vdp.setupNLP()
    vdp.solveNLP()
    vdp.simulateResults()
    vdp.plotSoln()
