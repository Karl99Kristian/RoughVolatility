import numpy as np
from scipy.linalg import expm

from RoughVolatility.scr.simulate import RSRBerbomi
from matplotlib import pyplot as plt
from RoughVolatility import DIR_PLOTS

lognorm = True
    
if __name__=="__main__":
    np.random.seed(2023)
    #### PARAMETERS

    # Time parameters for grid
    T = 3/252   
    dt = 3*252    
    U = T + 1/12    
    
    # \mu parameters 
    states = [0.62217679, 4.70115552]     
    qs = [0.65060997,3.4261729]                   
    Q = np.array([[0, qs[0]], [qs[1], 0]])  
    Q = Q - np.diag(np.sum(Q,axis=1))

    if lognorm:
        P = expm(Q/dt)
        dist = [P[1,0]/(P[0,1]+P[1,0]),P[0,1]/(P[0,1]+P[1,0])]
        m=dist[0]*states[0]+dist[1]*states[1]
        print(r"$E[\mu]= $",m)
        states = [m,m]
    

    # Model parameters
    theta = 5.84356644          
    H = 0.09610816
    alpha = H +1/2    
    nu = 0.27091626             
    eta = 0.53021322     
    xi_0 = 0.04263963 

    # Monte Carlo parameters
    N = 100000
    M = 2000   

    #### RUN DIRECT MODEL  

    # Initialize grids in rough Bergomi with stochastic MPVR
    mod = RSRBerbomi(T,dt,U,Q,states,theta,alpha,nu,eta, xi_0)

    # Initialize simulation for G, simulation of W on (0,t) and \mu on (0,u) 
    mod.init(N, M)

    # Calculate Xi for u\in(t+\Delta t, t + \Delta) 
    mod.calc_xis(N)
    
    # Calc the VIX
    urange = mod.timestampu[mod.nt:]
    vix_direct = np.zeros(N)

    temp = 0
    # Trapezoid
    for i, u in enumerate(urange[1:]):
        vix_direct += (mod.xis[i+1,:]+mod.xis[i,:])/2
        temp += 1/dt

    vix_direct = 12/dt*vix_direct 

    #### RUN DIRECT MODEL  
    mod = RSRBerbomi(T,dt,U,Q,states,theta,alpha,nu,eta, xi_0)

    s2_Y = mod.calc_var_Y()

    s2_M = mod.calc_var_M()

    s2 = (12*eta)**2*(nu**2*s2_Y+(1-nu**2)*s2_M)

    Ns = [100, 500, 750, 1000,5000,7500, 10000, 50000, 100000, 1000000]
    vix_approx = np.zeros((3,len(Ns)))
    for i,N in enumerate(Ns):
        np.random.seed(2023)
        mod.init_approx(N, M, s2)

        mod.simulate_approx_vix(N)

        vix_approx[0,i] = np.mean(mod.vix_approx)
        vix_approx[1,i] = np.var(mod.vix_approx)
        vix_approx[2,i] = np.mean((mod.vix_approx-vix_approx[0,i])**3)/(vix_approx[1,i])**(3/4)

    direct_mean =np.mean(vix_direct)
    direct_var =np.var(vix_direct)
    direct_skew = np.mean((mod.vix_approx-direct_mean)**3)/(direct_var)**(3/4)

    direct_moments = [direct_mean,direct_var,direct_skew]
    names = ["mean", "variance", "skewness"]

    bound_arr = [0.00175, 0.000075, 0.008]

    fig, ax = plt.subplots(ncols=3, figsize = (15,7))
    for i, n in enumerate(names):
        ax[i].plot(np.log(Ns),np.abs(direct_moments[i]-vix_approx[i,:]))
        ax[i].set_title(n)
        ax[i].set_ylim(0,bound_arr[i])

    fig.suptitle("Error on central moments")

    if lognorm:
        fig.savefig(DIR_PLOTS.joinpath("moment_error_jim.png"))
    else:
        fig.savefig(DIR_PLOTS.joinpath("moment_error.png"))
