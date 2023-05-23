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

def σ(x,_σ,σ_m):
    return np.sqrt(_σ**2 + (σ_m*x)**2)

def dσ(x,_σ,σ_m):
    return (σ_m**2*x)/np.sqrt(_σ**2 + (σ_m*x)**2)

def F_α(x,α):
    return α*x - x**3

def F_hat_α(x,α,_σ,σ_m):
    return F_α(x,α) + σ(x,_σ,σ_m)*dσ(x,_σ,σ_m)/2

def f_m(x,m,α,_σ,σ_m):
    return 2*integrate.quad(lambda y: (-F_hat_α(y,α,_σ,σ_m) + θ*(y-m))/(σ(y,_σ,σ_m)**2), 0, x)[0] + np.log(σ(x,_σ,σ_m)**2)

def f_m_det(x,m,α,_σ,σ_m):
    return -(α-θ-.5*σ_m**2+(_σ/σ_m)**2)/(σ_m**2)*np.log(1+(σ_m*x/_σ)**2) + (x/σ_m)**2 - 2*θ*m/(_σ*σ_m)*np.arctan(σ_m/_σ*x)

def ρ_st(x,θ,m,α,_σ,σ_m):
    if f'{θ},{m},{α},{_σ},{σ_m}' not in Zs.keys():
        Zs[f'{θ},{m},{α},{_σ},{σ_m}'] = integrate.quad(lambda y: np.exp(-f_m_det(y,m,α,_σ,σ_m)),-np.inf,np.inf)[0]
    Z = Zs[f'{θ},{m},{α},{_σ},{σ_m}']
    return np.exp(-f_m_det(x,m,α,_σ,σ_m))/Z

def R(θ,m,α,_σ,σ_m):
    return integrate.quad(lambda x: x*ρ_st(x,θ,m,α,_σ,σ_m), -np.inf, np.inf)[0]

if __name__=='__main__':

    #-----PARAMETERS-----
    α = 1
    θ = 4
    σ_m = 0.2
    _σ = 1.

    #-----LOOP FOR BIFURCATION SCHEME-----
    #Init
    σs = np.linspace(1,2,200)
    ms = [] 

    for _σ in σs:
        print(_σ)
        try:
            # α = α + σ_m*normal(0,.1)
            m_st = optimize.fsolve(lambda m: m - R(θ,m,α,_σ,σ_m), x0=_σ) 
        except RuntimeError:
            m_st = 0
        ms.append(m_st)

    #Plot
    plt.scatter(σs,ms)
    plt.show()
