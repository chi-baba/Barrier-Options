import numpy as np
from scipy.stats import norm

S0 = 100 # current price of the underlying
K = 110 # strike price
T = 1 # time to maturity
sigma = 0.3 # volatility
r = 0.03 # risk-free rate
q = 0.0 # dividend yield
B = 105 # barrier B > S 
R = 0.5 # rebate

def up_out_put_barrier(S0,K,T,r,q,sigma,B,R):
    phi, eta = -1, -1
    h = (r-q+0.5*sigma**2) / (sigma**2)
    mu = (r - q - 0.5*sigma**2)
    a = mu / (sigma**2) 
    b = np.sqrt(mu**2 + (2*r*sigma**2)) / (sigma**2)
    z = (np.log(B/S0) / (sigma*np.sqrt(T))) + (b*sigma*np.sqrt(T))
    x = (np.log(S0/K) / (sigma*np.sqrt(T))) + (h*sigma*np.sqrt(T))
    x1 = (np.log(S0/B) / (sigma*np.sqrt(T))) + (h*sigma*np.sqrt(T))
    y = (np.log((B**2) / (S0*K)) / (sigma*np.sqrt(T))) + (h * sigma * np.sqrt(T))  
    y1 = (np.log(B/S0) / (sigma*np.sqrt(T))) + (h * sigma * np.sqrt(T))
    v6 = R *(((B/S0)**(a+b))*norm.cdf(eta*z) + ((B/S0)**(a-b))*norm.cdf(eta*z - 2*eta*b*sigma*np.sqrt(T)))
    if K > B:
        v2 = phi*S0*np.exp(-q*T)*norm.cdf(phi*x1) - (phi*K*np.exp(-r*T)*norm.cdf(phi*x1-phi*sigma*np.sqrt(T)))
        v4 = phi*S0*np.exp(-q*T)*((B/S0)**(2*h))*norm.cdf(eta*y1) - \
            (phi*K*np.exp(-r*T)*((B/S0)**(2*h-2))*norm.cdf(eta*y1-eta*sigma*np.sqrt(T))) 
        val = v2 - v4 + v6
    else:
        v1 = phi*S0*np.exp(-q*T)*norm.cdf(phi*x) - (phi*K*np.exp(-r*T)*norm.cdf((phi*x)-(phi*sigma*np.sqrt(T))))
        v3 = phi*S0*np.exp(-q*T)*((B/S0)**(2*h))*norm.cdf(eta*y) - \
            (phi*K*np.exp(-r*T)*((B/S0)**(2*h-2))*norm.cdf(eta*y-eta*sigma*np.sqrt(T)))
        val = v1 - v3 + v6
    return val


prc = up_out_put_barrier(S0,K,T,r,q,sigma,B,R)
print('\nThe price of the up and out put option is:',round(prc,4))



