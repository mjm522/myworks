from casadi import *
import numpy as NP
import matplotlib.pyplot as plt

class vdpCollocation():
  def __init__(self):
    d = 3 # Degree of interpolating polynomial
    nk = 20 # Control discretization
    tf = 10.0 # End time
    h = tf/nk
    
    tau_root = [0] + collocation_points(d, "radau") # Choose collocation points
    C = NP.zeros((d+1,d+1)) # Coefficients of the collocation equation
    D = NP.zeros(d+1) # Coefficients of the continuity equation
    F = NP.zeros(d+1) # Coefficients of the quadrature function
    T = NP.zeros((nk,d+1)) # All collocation time points
    
    self.d = d
    self.nk = nk # Control discretization
    self.tf = tf # End time
    self.h = h # Size of the finite elements
    # Construct polynomial basis
    for j in range(d+1):
      # Construct Lagrange polynomials to get the polynomial basis at the collocation point
      p = NP.poly1d([1])
      for r in range(d+1):
        if r != j:
          p *= NP.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
        D[j] = p(1.0) # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        pder = NP.polyder(p)
        for r in range(d+1): # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
          C[j,r] = pder(tau_root[r])

          # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = NP.polyint(p)
        F[j] = pint(1.0)
      for k in range(nk):
        for j in range(d+1):
          T[k,j] = h*(k + tau_root[j])
      
      self.T = T; self.C = C; self.D = D; self.F= F;
     
  def problemSpecificParam(self): # Declare variables (use scalar graph)
    t  = SX.sym("t")    # time
    u  = SX.sym("u")    # control
    x  = SX.sym("x",2)  # state
    nx = 2 # Dimensions
    nu = 1
    
    return t, u, x, nx, nu
  
  def vdpODE(self): # ODE rhs function and quadratures
    t, u, x, _, _ = self.problemSpecificParam()
    xdot = vertcat((1 - x[1]*x[1])*x[0] - x[1] + u, x[0])
    qdot = x[0]*x[0] + x[1]*x[1] + u*u
    f = Function('f', [t,x,u],[xdot, qdot])
    
    return f

  def ctrlBounds(self): # Control bounds
    u_min = -0.75
    u_max = 1.0
    u_init = 0.0
    
    u_lb = NP.array([u_min])
    u_ub = NP.array([u_max])
    u_init = NP.array([u_init])
    
    return u_min, u_max, u_init, u_lb, u_ub, u_init
  
  def stateBounds(self): # State bounds and initial guess 
    x_min =  [-inf, -inf]
    x_max =  [ inf,  inf]
    xi_min = [ 0.0,  1.0]
    xi_max = [ 0.0,  1.0]
    xf_min = [ 0.0,  0.0]
    xf_max = [ 0.0,  0.0]
    x_init = [ 0.0,  0.0]
    
    return x_min, x_max, xi_min, xi_max, xf_min, xf_max, x_init

  def constraints(self):
    
    T = self.T; C = self.C; D = self.D; F = self.F;
    nk = self.nk; d = self.d; h = self.h; 
    t, u, x, nx, nu = self.problemSpecificParam();
    x_min, x_max, xi_min, xi_max, xf_min, xf_max, x_init = self.stateBounds()
    u_min, u_max, u_init, u_lb, u_ub, u_init = self.ctrlBounds()
    f = self.vdpODE()
    
    g = [] # Constraint function for the NLP
    lbg = []
    ubg = []
    
    NX = nk*(d+1)*nx        # Collocated states
    NU = nk*nu              # Parametrized controls
    NXF = nx                # Final state
    NV = NX+NU+NXF          # Total number of variables
    V = MX.sym("V",NV) # NLP variable vector
    
    vars_lb = NP.zeros(NV) # All variables with bounds and initial guess
    vars_ub = NP.zeros(NV)
    vars_init = NP.zeros(NV)
    offset = 0
    
    X = NP.resize(NP.array([],dtype=MX),(nk+1,d+1)) # Get collocated states and parametrized control
    U = NP.resize(NP.array([],dtype=MX),nk)
    
    for k in range(nk):        
      for j in range(d+1): # Collocated states
        X[k,j] = V[offset:offset+nx] # Get the expression for the state vector    
        vars_init[offset:offset+nx] = x_init # Add the initial condition
        if k==0 and j==0: # Add bounds
          vars_lb[offset:offset+nx] = xi_min
          vars_ub[offset:offset+nx] = xi_max
        else:
          vars_lb[offset:offset+nx] = x_min
          vars_ub[offset:offset+nx] = x_max
        offset += nx
        
      U[k] = V[offset:offset+nu] # Parametrized controls
      vars_lb[offset:offset+nu] = u_min
      vars_ub[offset:offset+nu] = u_max
      vars_init[offset:offset+nu] = u_init
      offset += nu

    X[nk,0] = V[offset:offset+nx] # State at end time
    vars_lb[offset:offset+nx] = xf_min
    vars_ub[offset:offset+nx] = xf_max
    vars_init[offset:offset+nx] = x_init
    offset += nx 
    
    J = 0 # Objective function 
    for k in range(nk): # For all finite elements
      for j in range(1,d+1): # For all collocation points
        xp_jk = 0 # Get an expression for the state derivative at the collocation point
        for r in range (d+1):
          xp_jk += C[r,j]*X[k,r]
          
        fk,qk = f(T[k,j], X[k,j], U[k]) # Add collocation equations to the NLP
        g.append(h*fk - xp_jk)
        lbg.append(NP.zeros(nx)) # equality constraints
        ubg.append(NP.zeros(nx)) # equality constraints

        J += F[j]*qk*h # Add contribution to objective

      xf_k = 0 # Get an expression for the state at the end of the finite element
      for r in range(d+1):
        xf_k += D[r]*X[k,r]
        
      g.append(X[k+1,0] - xf_k) # Add continuity equation to NLP
      lbg.append(NP.zeros(nx))
      ubg.append(NP.zeros(nx)) 
      
    g = vertcat(*g) # Concatenate constraints
    
    return V, J, g, vars_init, vars_lb, vars_ub, lbg, ubg
          
  def solveNLP(self): # SOLVE THE NLP
    V, J, g, vars_init, vars_lb, vars_ub, lbg, ubg = self.constraints()
    
    nlp = {'x':V, 'f':J, 'g':g}
    
    opts = {}  # Set options
    opts["expand"] = True
    #opts["ipopt.max_iter"] = 4
    solver = nlpsol("solver", "ipopt", nlp, opts)# Allocate an NLP solver
    
    arg = {}
    arg["x0"]  = vars_init # Initial condition
    arg["lbx"] = vars_lb # Bounds on x
    arg["ubx"] = vars_ub
    arg["lbg"] = NP.concatenate(lbg) # Bounds on g
    arg["ubg"] = NP.concatenate(ubg)

    res = solver(**arg) # Solve the problem
    print "optimal cost: ", float(res["f"]) # Print the optimal cost
    
    return res

  def retrieveSoln(self):
    res = self.solveNLP()
    nk = self.nk; d = self.d; h = self.h; tf = self.tf;
    t, u, x, nx, nu = self.problemSpecificParam();
    v_opt = NP.array(res["x"]) # Retrieve the solution
    self.x0_opt = v_opt[0::(d+1)*nx+nu] # Get values at the beginning of each finite element
    self.x1_opt = v_opt[1::(d+1)*nx+nu]
    self.u_opt = v_opt[(d+1)*nx::(d+1)*nx+nu]
    self.tgrid = NP.linspace(0,tf,nk+1)
    self.tgrid_u = NP.linspace(0,tf,nk)
    
    
  def plotSoln(self): # Plot the results
    x0_opt = self.x0_opt; x1_opt = self.x1_opt; u_opt = self.u_opt
    tgrid = self.tgrid; tgrid_u = self.tgrid_u
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid,x0_opt,'--')
    plt.plot(tgrid,x1_opt,'-.')
    plt.step(tgrid_u,u_opt,'-')
    plt.title("Van der Pol optimization")
    plt.xlabel('time')
    plt.legend(['x[0] trajectory','x[1] trajectory','u trajectory'])
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
  vdp = vdpCollocation()
  vdp.retrieveSoln()
  vdp.plotSoln() 