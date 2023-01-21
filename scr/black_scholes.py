import numpy as np
from scipy.stats import norm 
from scipy.optimize import brentq

def callPriceBS(r, spot, strike, sigma, tau):
    """
    Calculate call price in BS model
    """

    # Validate parameters
    if sigma<0 or tau < 0:
        return 9999

    d1 = 1/(sigma*np.sqrt(tau))*(np.log(spot/strike)+(r+sigma**2/2)*tau)
    d2 = d1 - sigma*np.sqrt(tau)
    return norm.cdf(d1)*spot-norm.cdf(d2)*strike*np.exp(-r*tau)

def impVol(r, price, spot, strike, tau):
    """
    Calculates implied volatility by bisection method
    """

    # Ensure price is at leas intrisic
    price = np.maximum(price, np.maximum(spot - strike, 0))

    def f(x):
        return price - callPriceBS(r=r,spot=spot,strike=strike,sigma=x,tau=tau)
    try:
        s2 = brentq(f, 1e-9, 1e+9)
    except ValueError:
        return 1e-9
    return s2







