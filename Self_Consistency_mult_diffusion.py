"""
Tries to solve self consistency equation on m with a multiplicative function sigma
"""
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from numpy.random import uniform
import matplotlib.pyplot as plt

Zs = {}

def σ(x):
    return np.sqrt(2/β)

def dσ(x):
    return 0

def F_α(x):
    return -(x**3 - x)

def F_hat_α(x):
    return F_α(x) + σ(x)*dσ(x)/2

def f_m(x,m):
    return 2*integrate.quad(lambda y: (-F_hat_α(y) + θ*(y-m))/(σ(y)**2), 0, x)[0] #+ np.log(σ(x)**2)

def ρ_st(x,m):
    if f'{β},{θ}' not in Zs.keys():
        Zs[f'{β},{θ}'] = integrate.quad(lambda y: f_m(y,m),-np.inf,np.inf)[0]
    Z = Zs[f'{β},{θ}']
    return f_m(x,m)/Z

def R(m):
    return integrate.quad(lambda x: x*ρ_st(x,m), -np.inf, np.inf)[0]

def R_prime(m):
    return integrate.quad(lambda x: β*θ*(x-m)*x*ρ_st(x,m), -np.inf, np.inf)[0]

if __name__=='__main__':
    #-----PARAMETERS-----
    # α = 1
    # β = 1
    # θ = 1 

    #taking different values of x_0 to see bifurcation 
    # Xs_0 = np.linspace(-1,1,30)

    # for x_0 in Xs_0:
    #     m_st = optimize.newton(lambda m: m - R(m), x0=x_0)
    #     print(m_st)

    xs = np.linspace(-1,1,10)
    β = 1
    θ = 1
    for x in xs:
        print(f_m(x,0))

    #PARAMETERS
    # θ = 1
    # βs = np.linspace(1,5,50)
    # ms = []
    # for β in βs:
    #     m_st = optimize.newton(lambda m: m - R(m), x0=np.random.choice([-.1,.1]),fprime=lambda m: 1 - R_prime(m))
    #     ms.append(m_st)

    # plt.scatter(βs,ms)
    # plt.show()
