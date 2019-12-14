import numpy as np
from scipy.stats import binom
import scipy.stats as stats
#from payoffs import VanillaOption, call_payoff, put_payoff
from collections import namedtuple

PricerResult = namedtuple('PricerResult', ['price', 'stderr'])

def european_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    
    for i in range(num_nodes):
        spot_t = spot * (u ** (steps - i)) * (d ** (i))
        call_t += option.payoff(spot_t) * binom.pmf(steps - i, steps, pstar)

    call_t *= np.exp(-rate * expiry)
    
    return call_t

def american_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    disc = np.exp(-rate * h) 
    spot_t = np.zeros(num_nodes)
    prc_t = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        spot_t[i] = spot * (u ** (steps - i)) * (d ** (i))
        prc_t[i] = option.payoff(spot_t[i])


    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            prc_t[j] = disc * (pstar * prc_t[j] + (1 - pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            prc_t[j] = np.maximum(prc_t[j], option.payoff(spot_t[j]))
                    
    return prc_t[0]

    
def naive_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    # return (prc_t[0], stderr)
    expiry = option.expiry
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    spot_t = np.empty(nreps)
    z = np.random.normal(size = nreps)

    for j in range(1, nreps):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps)

    return PricerResult(prc, se)

def antithetic_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    # return (prc_t[0], stderr)
    expiry = option.expiry
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    spot_t = np.empty(nreps * 2)
    z = np.random.normal(size = nreps)
    z_neg=-z
    X = np.concatenate((z, z_neg))

    for j in range(1, nreps * 2):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * X[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps *2)

    return PricerResult(prc, se)
  
def stratified_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    # return (prc_t[0], stderr)
    expiry = option.expiry
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    u = np.random.uniform(size = nreps)
    uhat = np.zeros(nreps)
    for i in range(nreps):
        uhat[i] = (i + u[i]) / nreps
    spot_t = np.empty(nreps)
    z = stats.norm.ppf(uhat)

    for j in range(1, nreps):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps)

    return PricerResult(prc, se)    
    

   #if __name__ == "__main__":
    #print("This is a module. Not intended to be run standalone.")
