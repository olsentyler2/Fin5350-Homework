import numpy as np
from scipy.stats import norm
from collections import namedtuple


PricerResult = namedtuple('PricerResult', ['price', 'stderr'])

def BlackScholesDelta(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - dividend + 0.5 * volatility * volatility) * tau) / (volatility * np.sqrt(tau))
    delta = np.exp(-dividend * tau) * norm.cdf(d1) 
    return delta

def naive_pricer(option, spot, rate, vol, div, steps, reps):
    pass

def BlackScholesGamma(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - dividend + 0.5 * volatility * volatility) * tau) / (volatility * np.sqrt(tau))
    gamma = np.exp(-dividend * tau) * norm.pdf(d1) / (spot * volatility * np.sqrt(tau))
    return gamma

def naive_pricer(option, spot, rate, vol, div, steps, reps):
    pass

def control_variate_pricer(option, spot, rate, vol, div, steps, reps, beta):
    expiry = option.expiry
    strike = option.strike
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigsdt = vol * np.sqrt(dt)
    erddt = np.exp((rate - div) * dt)    
    cash_flow_t = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_t = spot
        convar = 0.0
        z = np.random.normal(size=int(steps))

        for i in range(int(steps)):
            t = i * dt
            delta = BlackScholesDelta(spot, t, strike, expiry, vol, rate, div)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt)
            spot_t = spot_tn

        cash_flow_t[j] = option.payoff(spot_t) + beta * convar

    prc = np.exp(-rate * expiry) * cash_flow_t.mean()
    se = np.std(cash_flow_t, ddof=1) / np.sqrt(reps)
    
    return PricerResult(prc, se)

def control_variate_pricer_antithetic(option, spot, rate, vol, div, steps, reps, beta):
    expiry = option.expiry
    strike = option.strike
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigsdt = vol * np.sqrt(dt)
    erddt = np.exp((rate - div) * dt)    
    cash_flow_t = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_t = spot
        spot_t2 = spot
        convar = 0.0
        convar2 = 0.0
        z = np.random.normal(size=int(steps))

        for i in range(int(steps)):
            t = i * dt
            delta = BlackScholesDelta(spot_t, t, strike, expiry, vol, rate, div)
            delta2 = BlackScholesDelta(spot_t2, t, strike, expiry, vol, rate, div)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            spot_tn2 = spot_t2 * np.exp(nudt + sigsdt * z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt)
            convar2 = convar2 + delta * (spot_tn2 - spot_t2 * erddt)
            spot_t = spot_tn
            spot_t2 = spot_tn2

        #cash_flow_t[j] = option.payoff(spot_t) + beta * convar
        cash_flow_t[j] = 0.5 * (option.payoff(spot_t) + beta * convar + option.payoff(spot_t2) + beta * convar2)
        #cash_flow_t[j] = option.payoff(spot_t) + beta * convar

    prc = np.exp(-rate * expiry) * cash_flow_t.mean()
    se = np.std(cash_flow_t, ddof=1) / np.sqrt(reps)
    
    return PricerResult(prc, se)

def control_variate_pricer_antithetic_gamma(option, spot, rate, vol, div, steps, reps, beta):
    expiry = option.expiry
    strike = option.strike
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * dt
    sigsdt = vol * np.sqrt(dt)
    erddt = np.exp((rate - div) * dt)
    egamma = np.exp((2*(rate - div)+vol*vol)*dt)-2*erddt + 1
    cash_flow_t = np.zeros(reps)
    price = 0.0
    beta2 = -0.5

    for j in range(reps):
        spot_t = spot
        spot_t2 = spot
        convar = 0.0
        convar2 = 0.0
        z = np.random.normal(size=int(steps))

        for i in range(int(steps)):
            t = i * dt
            delta = BlackScholesDelta(spot_t, t, strike, expiry, vol, rate, div)
            delta2 = BlackScholesDelta(spot_t2, t, strike, expiry, vol, rate, div)
            gamma = BlackScholesGamma(spot_t, t, strike, expiry, vol, rate, div)
            gamma2 = BlackScholesGamma(spot_t2, t, strike, expiry, vol, rate, div)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            spot_tn2 = spot_t2 * np.exp(nudt + sigsdt * z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt) + delta2 * (spot_tn2 - spot_t2 * erddt)
            convar2 = convar2 + gamma * ((spot_tn - spot_t)*(spot_tn - spot_t) - spot_t * spot_t * egamma) + gamma2 * ((spot_tn2 - spot_t2) *(spot_tn2 -    spot_t2) - spot_t2*  spot_t2 * egamma)
            spot_t = spot_tn
            spot_t2 = spot_tn2

        #cash_flow_t[j] = option.payoff(spot_t) + beta * convar
        cash_flow_t[j] = 0.5 * (option.payoff(spot_t) + option.payoff(spot_t2) + beta * convar + beta2 * convar2)
        #cash_flow_t[j] = option.payoff(spot_t) + beta * convar

    prc = np.exp(-rate * expiry) * cash_flow_t.mean()
    se = np.std(cash_flow_t, ddof=1) / np.sqrt(reps)
    
    return PricerResult(prc, se)