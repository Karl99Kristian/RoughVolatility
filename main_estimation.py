import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored
from scipy import optimize
from datetime import datetime

from scr.black_scholes import impVol
from scr.simulate import RSRBerbomi
from RoughVolatility import DIR_DATA, DIR_PLOTS

# FLAG
optim = False

## Get data
fname = "VIX_calc"

# Cols are ttm, spot, obsdate, K, IV
arrx = np.loadtxt(DIR_DATA.joinpath(f"{fname}.csv"),dtype=float,delimiter=",",skiprows=1)

ttms = np.unique(np.sort(arrx[:,0]))

ttm_idx = 0

# Filter dates
arr = arrx[np.where(arrx[:,0]==ttms[ttm_idx])]


## Fixed Global params
# Contract params

T = ttms[ttm_idx]          # time to maturity of VIX option t
dt = 2 * 252             # timesteps pr year i.e. \Delta t = 1/"dt"
U = T + 1/12              # Max time of observed data. I.e t+\Delta where \Delta = 1 month

obsdate = arr[0,2]
spot = arr[0,1]
strikes = arr[:,3]
iVol_data = arr[:,4]

# Monte Carlo parameters
N = 10000   # Number of VIX simulations
M = 1000    # Number of simulations pr. starting point for estimation of G(\eta,u,s_0)

def plot_calib(mod_data, ext:bool=False):
    lm = -np.log(spot/strikes)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")

    ax.plot(lm, mod_data, label="Model fit", color = cmap(0))
    ax.scatter(lm, iVol_data, label="Real", color = cmap(1))
    ax.legend()
    ax.grid()
    
    ax.set_xlabel(r"logmoneyness, $\log\frac{K}{VIX_0}$")
    ax.set_ylabel("IV")

    # ax.set_ylim((0.5,3.25))

    ax.set_title(rf"date {obsdate:.0f}, t={T*252:.0f} Day(s), $\xi_0=${params[-1]:.04f}, $VIX^2_0=${spot**2}")

    outnm = f"obs_{obsdate:.0f}_ttm_{T*252:.0f}"
    if ext:
        outnm += "_exited"
    fig.savefig(DIR_PLOTS.joinpath(f"{outnm}.png"))
    return ax

def unpack_params(params):
    qs = [params[0],params[1]]                    # Transistion intesities
    Q = np.array([[0, qs[0]], [qs[1], 0]])        # Setup intesity matrix
    Q = Q - np.diag(np.sum(Q,axis=1))

    states = [params[2], params[3]]

    theta = params[4]
    alpha = params[5]
    nu = params[6]
    eta = params[7]
    xi_0 = params[8]

    return Q, states, theta, alpha, nu, eta, xi_0

def run_model(params):
    Q, states, theta, alpha, nu, eta, xi_0 = unpack_params(params)

    # Initialize grids in rough Bergomi with stochastic MPVR
    mod = RSRBerbomi(T,dt,U,Q,states,theta,alpha,nu,eta, xi_0)

    # Initialize simulation for G, simulation of W on (0,t) and \mu on (0,u) 
    mod.init(N, M)

    # Calculate Xi for u\in(t+\Delta t, t + \Delta) 
    mod.calc_xis(N)
    # Calc the VIX
    urange = mod.timestampu[mod.nt:]
    vix = np.zeros(N)

    # Trapezoid
    for i, _ in enumerate(urange[1:]):
        vix += (mod.xis[i-1,:]+mod.xis[i,:])/2
        
    vix = np.sqrt(12/dt*vix) 
    print(np.sort(vix))

    # Calc implied Vol by MC
    iVol = np.zeros(strikes.shape)
    prices = np.zeros(strikes.shape)
    for i, strike in enumerate(strikes):
        prices[i] = np.mean(np.maximum(vix-strike,0))
        iVol[i] = impVol(r=0, price=prices[i], spot=spot, strike=strike, tau=T)
        
    return iVol

def mse(params):
    iVol_mod = run_model(params)

    # Check iv
    res = 99999

    if np.min(iVol_mod)<1e-8:
        print("hit 0 iv")
        return res

    res = np.mean((iVol_mod-iVol_data)**2)
    print(colored(res, "green"),params)
    return res

if __name__=="__main__":
    print(f"Currently calibrating ttm = {T}, i.e. {T*252} days.")
    print(f"Timesteps of size {1/dt} i.e. {dt/252} steps pr day")
    print(f"Noice is evaluated on {dt*T} timepoints. Calculations at {21*dt/252} timepoints")

    params = [0.699,3.4365,0.1239, 4.8671, 5.9165 ,0.0938 +1/2 , 0.1373, 2*np.sqrt(0.1761), 0.0654]

    if ttm_idx == 0:
        # two steps pr day 
        # mse = 0.006626684214606622
        params = [1.00162185,5.95111787,0.37889369,10.50849076,1.23313109,0.58253937,0.32249655,0.49728621,0.03982713]
        # 5 steps pr day N = 10000
        # mse = 0.006839603277672641
        params = [1.00139431,5.95319743,0.46635431,10.51000929,1.312887,0.60066434,0.44640669,0.54031233,0.03927361]
    if ttm_idx == 1:
        # 2 steps pr day
        # mse = 0.008573932826208308
        params = [0.5620626,3.39176933,0.62596769,4.81243888,5.87835959,0.59243378,0.2341917,0.53045532,0.04287261]
        # 3 steps pr day N=10000
        # mse = 0.008733410948449549 
        params = [0.65060997,3.4261729,0.62217679,4.70115552,5.84356644,0.59610816,0.27091626,0.53021322,0.04263963]
    if ttm_idx == 2:
        # 2 steps pr day
        # mse = 0.0033541941182442686
        # params = [0.89356332,3.49600264,0.5346618,4.92216825,5.86081637,0.60901856,0.5328438,0.5620847,0.05194881]
        params = [0.91656876,3.55421589,0.50787917,4.88947702,5.81879203,0.60050493,0.52350871,0.54277767,0.05270626]
        # 2 steps pr day N=10000
        # mse = 0.002229616100312466
        params = [0.92920166,3.58288378,0.50653214,4.83608877,5.74667305,0.60100455,0.53086677,0.53368318,0.05323128]
    if ttm_idx == 3:
        # 2 steps pr day N = 5000
        # mse = 0.002363673951069787
        params = [0.86059091,3.1910418,0.46568322,4.58618978,5.93776665,0.60046418,0.3875243,0.70554273,0.05271156]

    #### OPTIMIZE
    itercount = 0
    mse_spot = 20
    t1=datetime.now()

    if optim:
        try:
            while mse_spot >0.01:
                bnds=((0.0,2.0),(0.0,15.0),(0.0,1.0),(0.0,20.0),(0.1,10.0),(0.57,0.63),(0.01,0.99),(0.2,0.99),(0.0001,0.25))
                min_param = optimize.minimize(mse,params, method = "Trust-constr", bounds = bnds,options = {"disp":True, "maxiter":100})
                # min_param = optimize.minimize(mse,params, method = "Nelder-Mead", bounds = bnds,options = {"disp":True, "maxiter":100})
                # min_param = optimize.minimize(mse,params, method = "Nelder-Mead",options = {"disp":True, "maxiter":100})
                params = np.array(min_param.x)
                mse_spot = min_param.fun
                itercount += 100
                print(f"iterations: {itercount}")
                print(f"mse: {mse_spot}")
                
            print(mse_spot)
            print(params)
        finally:
            t2 = datetime.now()
            print(t2-t1)
            print(mse_spot)
            print(params)
            iVol_mod = run_model(params)
            plot_calib(iVol_mod, True)

    iVol_mod = run_model(params)
    plot_calib(iVol_mod)
        