import numpy as np
from datetime import datetime
from multiprocessing import Pool

from scipy.linalg import expm
from RoughVolatility.scr.mittag_leffler_master.mittag_leffler import ml
from scipy.special import gamma

class Model:
    """
    General model to setup timegrid and simulate Gaussian
    """
    def __init__(self,  T:int, dt: int) -> None:
        self.T = T
        self.dt = dt
        self.nt = int(T*dt+1)
        self.timestamp = np.arange(0,T+1/dt,1/dt)

    def simulateNormal(self, N: int, dim: int=1) -> None:
        if dim == 1:
            self.brownian = 1/np.sqrt(self.dt)*np.random.normal(size=(self.nt-1,N))
        else:
            self.brownian = 1/np.sqrt(self.dt)*np.random.normal(size=(dim,self.nt-1,N))


class MarkovChain(Model):
    """
    Model for simulating discrete markov chain
    """
    def __init__(self,T,dt,Q, states) -> None:
        super().__init__(T,dt)
        self.Q = Q
        self.states = states
    
    def simulate(self, N:int, start:int, void:bool=True):
        P = expm(self.Q/self.dt)
        states_idx = range(self.Q.shape[0])
        statesdict = {i: self.states[i] for i in states_idx}
        paths = np.zeros((self.nt,N))+start

        for i in range(1,self.nt):
            temp_pth = np.zeros(N)
            for j in states_idx:
                idx = np.argwhere(paths[i-1,:]==j)
                temp_pth[idx] = np.random.choice(states_idx,size=(1,len(idx)),p=P[j,:]).T
            paths[i,:] = temp_pth
        
        if void:
            self.path = np.vectorize(statesdict.__getitem__)(paths)
        else:
            return np.vectorize(statesdict.__getitem__)(paths)

class RSRBerbomi(Model):
    """
    Specialized model for the generalized bergomi 
    with fractional kernel and regime based MPVR
    """
    def __init__(self,T,dt, U,Q, states, theta, alpha, nu, eta, xi_0) -> None:
        super().__init__(T,dt)
        
        # U related grid
        self.U = U
        self.ntu = int(U*dt+1)
        self.timestampu = np.arange(0,U+1/dt,1/dt)#/self.dt
        
        # Parameters
        self.Q = Q
        self.states = states
        self.theta = theta
        self.alpha = alpha
        self.nu = nu
        self.eta = eta
        self.xi_0 = xi_0

        # Initialize \mu
        self.mc = MarkovChain(self.T,self.dt,self.Q,self.states)

        # Helper
        self.process = np.zeros((15,len(self.timestampu[self.nt:])))
    
    def kernel(self, x):
        return x**(self.alpha-1)

    def kernel_e(self, x):
        return x**(self.alpha-1)*gamma(self.alpha)*ml(-self.theta*gamma(self.alpha)*(x**self.alpha), self.alpha, self.alpha)

    def etu(self, u_idx, nu, t:bool):
        ts = 0
        if t:
            ts = self.nt-1
        
        temp = 0

        # Integrate over singularity by Bennedsen et al 2017
        hurst2 = 2*self.alpha-1

        appx = (1+(gamma(self.alpha)*ml(-self.theta*gamma(self.alpha)/(self.dt**self.alpha), self.alpha, self.alpha))**2)/2 
        analy = (1/self.dt)**hurst2/hurst2
        temp = appx*analy

        # Riemann approx (trapez) over rest
        if not(t) or u_idx>self.nt:
            tr = self.timestampu[2:u_idx-ts+1]
            for s in tr:
                temp += (self.kernel_e(s-1/self.dt)**2+self.kernel_e(s)**2)/2*1/self.dt
                
        temp *= (nu**2)*1/2
        return temp

    def cgf_m(self, u, nu, t:bool):
        ts = 0
        if t:
            ts = self.T
        hurst2 = 2*self.alpha -1
        return 1/2*(1-nu**2)*(u-ts)**hurst2/hurst2

    def get_integs(self, u_idx, u):
        integs = ml(-self.theta*gamma(self.alpha)*(u-self.timestampu[:u_idx+1])**self.alpha,self.alpha)
        self.integs = integs[1:]-integs[:-1]

    def est_G(self, u_idx, curr_states, t:bool):
        ts_idx = 0
        zs = [self.states[0]]
        if t:
            ts_idx = self.nt-1
            zs = self.states
        
        ests=np.zeros(len(curr_states))

        # Loop over states and evaluate estimate from paths starting in that state
        # We use time homogeneity in stead of markov property
        # (Former is nice in code since integs can be calc once, latter is nice in latex)
        for state_idx, state in enumerate(zs):  
            est = np.matmul(self.g_paths[state_idx,ts_idx:u_idx,:].T,self.integs[ts_idx:u_idx])          
            est = np.mean(np.exp(self.eta*est))
            ests[np.where(curr_states==state)] = est

        return ests

    def calc_voltarre(self,N, func, x, bmidx):
        temp = np.zeros(N)

        # Calc trapezoid
        for i,t in enumerate(self.timestamp[1:]):
            temp = temp + (func(x-t)+func(x-self.timestamp[i]))/2*self.brownian[bmidx,i,:]
        return temp

    def init(self, N, M):
        # Initialize everything that can be done outside loop

        # Simulate BM's
        self.simulateNormal(N,2)

        # Simulate MC for G
        mcg = MarkovChain(self.U,self.dt,self.Q,self.states)
        self.g_paths = np.zeros((len(self.states),mcg.nt,M))
        for state_idx, _ in enumerate(self.states):
            self.g_paths[state_idx]=mcg.simulate(M, state_idx,False)

        # Simulate MC for path
        self.mc.simulate(N,0)

    def calc_mu_dep(self, N, u_idx,u):
        # Calc integrals for H and G
        self.get_integs(u_idx, u)
        
        # Estimate G_0
        g_0=self.est_G(u_idx,np.ones(N)*self.states[0],False)
        # Estimate G_t
        g_t=self.est_G(u_idx,self.mc.path[-1,:],True)

        # Calc H
        H = np.matmul(self.mc.path[:self.nt-1,:].T,self.integs[:self.nt-1])
    
        return g_0, g_t, H

    def calc_gaussian(self, N, u):
        # Calc M
        M = self.calc_voltarre(N, self.kernel,u, 0)

        # Calc Y
        Y = self.calc_voltarre(N, self.kernel_e,u, 1)

        return M, Y

    def simulate_xi_direct(self, N, u_idx, u):
        # This function also records processes

        # Calc G and H
        g_0,g_t,H = self.calc_mu_dep(N,u_idx,u)

        # Calc M, Y
        M, Y = self.calc_gaussian(N,u)

        # Calc Lambda(0,t,u)
        Lamb = H+self.nu*Y+np.sqrt(1-self.nu**2)*M
        # Calc lambda(0,t,u)
        lamb = self.etu(u_idx,self.nu,True)-self.etu(u_idx,self.nu,False)+self.cgf_m(u,self.nu,True)-self.cgf_m(u,self.nu,False)

        # Record processes for tests        
        self.process[0,u_idx-self.nt]=np.mean(H)
        self.process[1,u_idx-self.nt]=np.mean(Y)
        self.process[2,u_idx-self.nt]=np.mean(M)
        self.process[3,u_idx-self.nt]=np.var(H)
        self.process[4,u_idx-self.nt]=np.var(Y)
        self.process[5,u_idx-self.nt]=np.var(M)
        self.process[6,u_idx-self.nt]=np.mean(g_0)
        self.process[7,u_idx-self.nt]=np.mean(g_t)
        self.process[8,u_idx-self.nt]=self.etu(u_idx,self.nu,True)
        self.process[9,u_idx-self.nt]=self.etu(u_idx,self.nu,False)
        self.process[10,u_idx-self.nt]=self.cgf_m(u,self.nu,True)
        self.process[11,u_idx-self.nt]=self.cgf_m(u,self.nu,False)
        self.process[12,u_idx-self.nt]=self.xi_0*np.exp(self.eta*(self.nu*Y+np.sqrt(1-self.nu**2)*M)+self.eta**2*lamb)[0]

        def mean_mu_s(s):
            P = expm(self.Q*s)
            return self.states[0]*P[0,0]+self.states[1]*P[1,0]

        means = np.zeros_like(self.timestamp[:self.nt-2])
        for i, s in enumerate(self.timestamp[:self.nt-2]):
            means[i] = mean_mu_s(s+1/self.dt)
        self.process[13,u_idx-self.nt]=np.matmul(means.T,self.integs[:self.nt-2])
        # print("mean mc at t",self.mc.path[-1,0])
        self.process[14,u_idx-self.nt]=np.var(g_t/g_0*np.exp(self.eta*H))

        return self.xi_0*g_t/g_0*np.exp(self.eta*Lamb+self.eta**2*lamb)

    def simulate_xi_direct_par(self, input):
        # unpack input
        N = int(input[0])
        i_idx = int(input[1])
        u_idx = int(i_idx+self.nt)
        u = input[2]

        # Calc G and H
        g_0,g_t,H = self.calc_mu_dep(N,u_idx,u)

        # Calc M, Y
        M, Y = self.calc_gaussian(N,u)

        # Calc Lambda(0,t,u)
        Lamb = H+self.nu*Y+np.sqrt(1-self.nu**2)*M
        # Calc lambda(0,t,u)
        lamb = self.etu(u_idx,self.nu,True)-self.etu(u_idx,self.nu,False)+self.cgf_m(u,self.nu,True)-self.cgf_m(u,self.nu,False)

        return self.xi_0*g_t/g_0*np.exp(self.eta*Lamb+self.eta**2*lamb)

    def calc_xis(self, N, rec:bool=False):
        urange = self.timestampu[self.nt:]
        data_inputs = zip(np.ones_like(urange)*N,np.arange(len(urange)),urange)

        self.xis = np.zeros((len(urange),N))

        t1 = datetime.now()

        # This takes 1.5 minutes, multiprocessing takes 17 sec.
        if rec:
            for i,u in enumerate(urange):
                self.xis[i,:] = self.simulate_xi_direct(N,i+self.nt, u)
        else:
            pool = Pool()
            res = pool.map(self.simulate_xi_direct_par, data_inputs) 
            for i,_ in enumerate(urange):
                self.xis[i,:]=res[i]

        t2 = datetime.now()
        
        print("XI Calc Time: ", t2-t1)
        
    def calc_m(self, N, u_idx, u):
        # Calc G and H
        g_0,g_t,H = self.calc_mu_dep(N,u_idx,u)

        lamb = self.etu(u_idx,self.nu,True)-self.etu(u_idx,self.nu,False)+self.cgf_m(u,self.nu,True)-self.cgf_m(u,self.nu,False)
        
        return self.eta**2*lamb+np.log(self.xi_0) +np.log(g_t/g_0)+self.eta*H 

    def calc_m_par(self, input):
        N = int(input[0])
        i_idx = int(input[1])
        u_idx = int(i_idx+self.nt)
        u = input[2]
        # Calc G and H
        g_0,g_t,H = self.calc_mu_dep(N,u_idx,u)

        lamb = self.etu(u_idx,self.nu,True)-self.etu(u_idx,self.nu,False)+self.cgf_m(u,self.nu,True)-self.cgf_m(u,self.nu,False)
        
        return self.eta**2*lamb+np.log(self.xi_0)+np.log(g_t/g_0)+self.eta*H 

    def calc_var_M(self):
        #calc last integral
        def f(x):
            return (x**2+x*1/12)**self.alpha
        
        integ = 0
        for i,t in enumerate(self.timestamp[1:]):
            integ += (f(self.timestamp[i])+f(t))/2
        integ *= 1/self.dt

        const = 2*self.alpha+1

        return 1/self.alpha**2*(1/const*((self.T+1/12)**const-1/12**const+self.T**const)-2*integ)

    def calc_var_Y(self):
        integs = ml(-self.theta*gamma(self.alpha)*(self.T-self.timestamp)**self.alpha,self.alpha)
        integs -= ml(-self.theta*gamma(self.alpha)*(self.T-self.timestamp+1/12)**self.alpha,self.alpha)
        
        temp = 0
        for i, I in enumerate(integs[1:]):
            temp += (I**2+integs[i]**2)/2
        temp = temp*1/self.dt*1/self.theta**2
        return temp

    def init_approx(self, N, M, var):
        # Simulate gaussian from var
        self.gaussian = np.sqrt(var)*np.random.normal(0,1,N)
        
        # Simulate MC for G
        mcg = MarkovChain(self.U,self.dt,self.Q,self.states)
        self.g_paths = np.zeros((len(self.states),mcg.nt,M))
        for state_idx, _ in enumerate(self.states):
            self.g_paths[state_idx]=mcg.simulate(M, state_idx,False)
        
        # Simulate MC for path
        self.mc.simulate(N,0)

    def simulate_approx_vix(self, N):
        # Calculate \overline{m}(u)
        urange = self.timestampu[self.nt:]
        data_inputs = zip(np.ones_like(urange)*N,np.arange(len(urange)),urange)
        self.m = np.zeros((len(urange),N))
        t1 = datetime.now()
        
        # # This takes 1.5 minutes, multiprocessing takes 17 sec.
        # for i,u in enumerate(urange):
        #     self.m[i,:] = self.calc_m(N,i+self.nt, u)#-np.log(self.xi_0)
        pool = Pool()
        res = pool.map(self.calc_m_par, data_inputs) 
        for i,_ in enumerate(urange):
            self.m[i,:]=res[i]
        # Trapezoid
        mean = np.zeros(N)
        temp=0
        for i,u in enumerate(urange[1:]):
            mean += (self.m[i,:] + self.m[i+1,:])/2
            temp += 1/self.dt

        # Should be normalized by Delta but we do trapez so we have one point less
        mean = mean*1/self.dt*1/temp

        self.vix_approx = np.exp((mean+self.gaussian))#*self.xi_0

        t2 = datetime.now()
        
        print("VIX Calc Time: ", t2-t1)
        
