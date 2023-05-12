"""
Tries to solve self consistency equation on m with a multiplicative function sigma
"""
import numpy as np
from numpy.random import normal
from scipy import integrate
import scipy.optimize as optimize
from numpy.random import uniform
import matplotlib.pyplot as plt

Zs = {} #Dict used for keeping the integration constant in the memory

def V(x):
    return x**4/4 - x**2/2

def dV(x):
    return x**3 - x

def d2V(x):
    return 3*x**2 - 1

def Φ(x,m,θ):
    return V(x) + .5*θ*(x-m)**2

def gradΦ(x,m,θ):
    return dV(x) + θ*(x-m)

def H(x,θ):
    return d2V(x) + θ

def f_m(x,m,D,τ,θ):
    return 1/D*(Φ(x,m,θ) + τ/2*gradΦ(x,m,θ)**2)               

def ρ_st(x,m,D,τ,θ):
    if f'{m},{D},{τ},{θ}' not in Zs.keys():
        Zs[f'{m},{D},{τ},{θ}'] = integrate.quad(lambda y: np.exp(-f_m(y,m,D,τ,θ))*np.abs(1 + τ*H(y,θ)),-np.inf,np.inf)[0]
    Z = Zs[f'{m},{D},{τ},{θ}']
    return np.exp(-f_m(x,m,D,τ,θ))*np.abs(1 + τ*H(x,θ))/Z

def R(m,D,τ,θ):
    return integrate.quad(lambda x: x*ρ_st(x,m,D,τ,θ), -np.inf, np.inf)[0]

def bifurcation_scheme():
    #PARAMETERS
    θ = 1
    τ = 1
    Ds = np.linspace(.23,1,100)
    βs = np.linspace(.5,5,200)
    ms = []
    for β in βs:
        D = 1/β
        try:
            m_st = optimize.newton(lambda m: m - R(m,D,τ,θ), x0=np.random.choice([-5*D,5*D]))
        except RuntimeError:
            m_st = 0
        ms.append(m_st)

    plt.scatter(βs,ms)
    plt.show()
    return ms

if __name__=='__main__':

    # D = 1
    # θ = 1
    # τ = 1

    # xs = np.linspace(-2,2,100)
    # plt.plot(xs,ρ_st(xs,1,D,τ,θ))
    # plt.show()
    bifurcation_scheme()