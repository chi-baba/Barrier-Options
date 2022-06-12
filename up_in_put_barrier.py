import numpy as np
from scipy.stats import norm

S0 = 100 # current price of the underlying
K = 95 # strike price
T = 1 # time to maturity
sigma = 0.3 # volatility
r = 0.03 # risk-free rate
q = 0.0 # dividend yield
B = 105 # barrier B > S 
R = 0.5 # rebate

def up_in_put_barrier(S0,K,T,r,q,sigma,B,R):
    phi, eta = -1, -1
    h = (r-q+0.5*sigma**2) / (sigma**2)
    x = (np.log(S0/K) / (sigma*np.sqrt(T))) + (h*sigma*np.sqrt(T))
    x1 = (np.log(S0/B) / (sigma*np.sqrt(T))) + (h*sigma*np.sqrt(T))
    y = (np.log((B**2) / (S0*K)) / (sigma*np.sqrt(T))) + (h * sigma * np.sqrt(T))  
    y1 = (np.log(B/S0) / (sigma*np.sqrt(T))) + (h * sigma * np.sqrt(T))
    v5 = R*np.exp(-r*T) * (norm.cdf(eta*x1-eta*sigma*np.sqrt(T)) - \
                           ((B/S0)**(2*h-2)) * norm.cdf(eta*y1-eta*sigma*np.sqrt(T))) 
    if K > B:
        v1 = phi*S0*np.exp(-q*T)*norm.cdf(phi*x) - (phi*K*np.exp(-r*T)*norm.cdf((phi*x)-(phi*sigma*np.sqrt(T))))
        v2 = phi*S0*np.exp(-q*T)*norm.cdf(phi*x1) - (phi*K*np.exp(-r*T)*norm.cdf(phi*x1-phi*sigma*np.sqrt(T)))
        v4 = phi*S0*np.exp(-q*T)*((B/S0)**(2*h))*norm.cdf(eta*y1) - \
            (phi*K*np.exp(-r*T)*((B/S0)**(2*h-2))*norm.cdf(eta*y1-eta*sigma*np.sqrt(T))) 
        val = v1 - v2 + v4 + v5
    else:
        v3 = phi*S0*np.exp(-q*T)*((B/S0)**(2*h))*norm.cdf(eta*y) - \
            (phi*K*np.exp(-r*T)*((B/S0)**(2*h-2))*norm.cdf(eta*y-eta*sigma*np.sqrt(T)))
        val = v3 + v5
    return val


prc = up_in_put_barrier(S0,K,T,r,q,sigma,B,R)
print('\nThe price of the up and in put option is:',round(prc,4))

# Monte Carlo Simulation
def up_in_put_barrier_mc(S0,K,T,r,q,sigma,B,R):
    N = 50000 # number of simulations
    M = 5000 # number of discrete steps in a price path
    dt = T/M # time length of steps
    wiener = np.random.normal(loc=0.0, scale=np.sqrt(dt), size = (N,M))
    St = S0 * np.exp(np.cumsum((r-q-0.5*sigma**2)*dt + sigma*wiener, axis=1))
    payoff = np.maximum(K-St[:,-1], 0)
    prc_max = np.amax(St, axis=1)
    payoff[prc_max <= B] = R
    prc = np.exp(-r*T) * payoff.mean()
    return prc

prc_mc = up_in_put_barrier_mc(S0,K,T,r,q,sigma,B,R)
print('\nThe Monte Carlo price of the up and in put option is:',round(prc_mc,4))
