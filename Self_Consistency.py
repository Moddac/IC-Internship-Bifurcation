"""
Tries to solve self consistency equation on m
"""
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

def V(x):
    return x**4/4 - x**2/2

def ρ_st(x,m):
    f = lambda y: np.exp(-β*(V(y)+.5*θ*(y-m)**2))
    try:
        Z
    except:
        Z = integrate.quad(f,-np.inf,np.inf)
    return f(x)/Z[0]

def R(m):
    return integrate.quad(lambda x: x*ρ_st(x,m), -np.inf, np.inf)[0]

if __name__=='__main__':
    β = 2.5
    θ = 1
    N = 10_000
    dt = .01
    N_p = 1_000

    #taking different values of x_0 to see bifurcation 
    Xs_0 = np.linspace(-1,1,30)

    for x_0 in Xs_0:
        m_st = optimize.newton(lambda m: m - R(m), x0=x_0)
        print(m_st)
