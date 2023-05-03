"""
Tries to solve self consistency equation on m
"""
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from numpy.random import uniform
import matplotlib.pyplot as plt

Zs = {}

def V(x):
    return x**4/4 - x**2/2

def ρ_st(x,m,β,θ):
    f = lambda y: np.exp(-β*(V(y)+.5*θ*(y-m)**2))
    if f'{β},{θ}' not in Zs.keys():
        Zs[f'{β},{θ}'] = integrate.quad(f,-np.inf,np.inf)[0]
    Z = Zs[f'{β},{θ}']
    return f(x)/Z

def R(m,β,θ):
    return integrate.quad(lambda x: x*ρ_st(x,m,β,θ), -np.inf, np.inf)[0]

def R_prime(m,β,θ):
    return integrate.quad(lambda x: β*θ*(x-m)*x*ρ_st(x,m,β,θ), -np.inf, np.inf)[0]

def bifurcation_scheme():
    #PARAMETERS
    θ = 1
    βs = np.linspace(1,5,200)
    ms = []
    for β in βs:
        m_st = optimize.newton(lambda m: m - R(m,β,θ), x0=np.random.choice([-.15*β,.15*β]),fprime=lambda m: 1 - R_prime(m,β,θ))
        ms.append(m_st)

    plt.scatter(βs,ms)
    plt.show()
    return ms

if __name__=='__main__':

    #taking different values of x_0 to see bifurcation 
    # Xs_0 = np.linspace(-1,1,30)

    xs = np.linspace(-1,1,10)
    β = 1
    θ = 1
    for x in xs:
        print(f_m(x,0))

    # for x_0 in Xs_0:
    #     m_st = optimize.newton(lambda m: m - R(m), x0=x_0)
    #     print(m_st)

    # bifurcation_scheme()
