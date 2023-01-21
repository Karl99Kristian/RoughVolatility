import numpy as np
from scipy.linalg import expm
from RoughVolatility.scr.simulate import RSRBerbomi
from RoughVolatility.scr.mittag_leffler_master.mittag_leffler import ml 
from scipy.special import gamma



    
if __name__=="__main__":
    np.random.seed(2024)
    #### FLAGS

    #### PARAMETERS

    # Time parameters for grid
    T = 3*21/252    
    dt = 1*252      
    U = T + 1/12    
    
    # \mu parameters 
    states = [0.1239, 4.8671]               
    qs = [0.699,13.4365]                   
    Q = np.array([[0, qs[0]], [qs[1], 0]])  
    Q = Q - np.diag(np.sum(Q,axis=1))

    # Model parameters
    theta = 5.9165          
    H = 0.0938
    alpha = 0.0938 +1/2    
    nu = 0.1373             
    eta = 2*np.sqrt(0.1761)     
    xi_0 = 0.0654           

    # Monte Carlo parameters
    N = 5000   
    M = 2000   

    #### RUN MODEL  

    # Initialize grids in rough Bergomi with stochastic MPVR
    mod = RSRBerbomi(T,dt,U,Q,states,theta,alpha,nu,eta, xi_0)

    # Initialize simulation for G, simulation of W on (0,t) and \mu on (0,u) 
    mod.init(N, M)

    # Calculate Xi for u\in(t+\Delta t, t + \Delta) 
    mod.calc_xis(N)
    xi = mod.xis
    mod.calc_xis(N,True)
    
    # Calc the VIX
    urange = mod.timestampu[mod.nt:]
    vix = np.zeros(N)

    # Trapezoid
    for i, u in enumerate(urange[1:]):
        vix += (mod.xis[i-1,:]+mod.xis[i,:])/2
        
    vix = np.sqrt(12/dt*vix) 


    ### TESTS
    ## Timestamp tests
    # timestamps to t is 1 + steps pr day * months * days in month
    assert len(mod.timestamp) == 1*3*21 + 1
    # timestamps to u_max is steps pr day * 1 month extra
    assert len(mod.timestampu) == len(mod.timestamp) + 1*21
    # time between timestamps is steps pr day
    assert (mod.timestamp[1:]-mod.timestamp[:-1]-1/dt<1e-5).all()
    
    ## Markov Chains
    # Q is intesity matrix
    assert (np.sum(Q, axis=1)<1e-5).all()
    P = expm(Q/dt)
    # P is probability matrix
    assert (np.sum(P,axis=1)-1<1e-5).all()
    dist = [P[1,0]/(P[0,1]+P[1,0]),P[0,1]/(P[0,1]+P[1,0])]
    p1 = np.mean((mod.g_paths[1,-1,:]==states[0]))
    p1 = (p1+np.mean((mod.g_paths[0,-1,:]==states[0])))/2
    # Emperical dist is equal to theoretical
    # No assert since U might not be that long run
    print("Limit dist emp = ", [p1,1-p1])
    print("Limit dist thr = ", dist)

    ## Calculations
    # xi are same with loop and multiprocess
    assert (xi - mod.xis<1e-5).all()

    # mean of M
    print("\nThese should be somewhat the same i think...")
    print(mod.process[13,:])
    print(mod.process[0,:])
    print()
    
    # mean of Y and M
    assert (mod.process[1,:]<1e-3).all() # Y
    assert (mod.process[2,:]<1e-1).all() # M
    print("Quote of mean M greater then 1e-2: ",np.sum(mod.process[2,:]>1e-2)/mod.process[2,:].size)
    
    # var of Y and M
    def psi(x):
        return x**(alpha-1)*gamma(alpha)*ml(-theta*gamma(alpha)*(x**alpha), alpha, alpha)
    def y_var(u):
        temp = np.zeros_like(u)
        for i,x in enumerate(mod.timestamp[:-1]):
            temp += (psi(u-x)**2+psi(u-mod.timestamp[i+1])**2)/2*1
        return temp*1/dt
    assert (mod.process[4,:]-y_var(urange)<1e-2).all()

    def m_var(u,t,H):
        return 1/(2*H)*(u**(2*H)-(u-t)**(2*H))
    assert (mod.process[5,:]-m_var(urange,T,H)<1e-1).all()

    ### Test also when intesities are zero and zero mean rev, or all states equal 
    # then G's and H should cancel
    states = [1.0, 1.0]
     # Initialize grids in rough Bergomi with stochastic MPVR
    mod = RSRBerbomi(T,dt,U,Q,states,theta,alpha,nu,eta, xi_0)

    # Initialize simulation for G, simulation of W on (0,t) and \mu on (0,u) 
    mod.init(N, M)

    # Calculate Xi for u\in(t+\Delta t, t + \Delta) 
    mod.calc_xis(N,True)

    print(mod.xis[:,0]-mod.process[12,:])

    print(mod.process[13,:])

    assert (mod.xis[:,0]-mod.process[12,:]<1e-3).all()