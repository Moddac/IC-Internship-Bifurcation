"""
Find the critical value of sigma  for an IPS with sigma(x) = sqrt(sigma**2 + (sigma_m*x)**2)
Case sigma_m = 0 is white noise
"""
import numpy as np
from numpy.random import normal
from scipy import integrate
import scipy.optimize as optimize
from numpy.random import uniform
import matplotlib.pyplot as plt

Zs = {} #Dict used for keeping the integration constant in the memory

"""
For multiplicative noise
def σ(x,_σ,σ_m):
    return np.sqrt(_σ**2 + (σ_m*x)**2)

def dσ(x,_σ,σ_m):
    return (σ_m**2*x)/np.sqrt(_σ**2 + (σ_m*x)**2)

def V_α(x, α):
    return x**4/4 - α*x**2/2

def F_α(x,α):
    return α*x - x**3

def F_hat_α(x,α,_σ,σ_m):
    return F_α(x,α) + σ(x,_σ,σ_m)*dσ(x,_σ,σ_m)/2
def f_m(x,m,α,θ,_σ,σ_m):
    if σ_m==0:
        return f_m_det(x,m,α,_σ, σ_m)
    return 2*integrate.quad(lambda y: (-F_hat_α(y,α,_σ,σ_m) + θ*(y-m))/(σ(y,_σ,σ_m)**2), 0, x)[0] + np.log(σ(x,_σ,σ_m)**2)

def f_m_det(x,m,α,_σ,σ_m):
    if σ_m==0:
        return 2/(_σ**2)*(V_α(x, α) + θ/2*(x-m)**2)
    return -(α-θ-.5*σ_m**2+(_σ/σ_m)**2)/(σ_m**2)*np.log(1+(σ_m*x/_σ)**2) + (x/σ_m)**2 - 2*θ*m/(_σ*σ_m)*np.arctan(σ_m/_σ*x)

def ρ_st(x,θ,m,α,_σ,σ_m):
    if f'{θ},{m},{α},{_σ},{σ_m}' not in Zs.keys():
        Zs[f'{θ},{m},{α},{_σ},{σ_m}'] = integrate.quad(lambda y: np.exp(-f_m_det(y,m,α,_σ,σ_m)),-np.inf,np.inf)[0]
    Z = Zs[f'{θ},{m},{α},{_σ},{σ_m}']
    return np.exp(-f_m_det(x,m,α,_σ,σ_m))/Z

def R(θ,m,α,_σ,σ_m):
    return integrate.quad(lambda x: x*ρ_st(x,θ,m,α,_σ,σ_m), -np.inf, np.inf)[0]

def R_prime_zero(θ,α,_σ,σ_m):
    if σ_m==0:
        return 2*θ/(_σ**2)*integrate.quad(lambda x: x**2*ρ_st(x,θ,0,α,_σ,σ_m), -np.inf, np.inf)[0]
    else: 
        return 2*θ/(_σ*σ_m)*integrate.quad(lambda x: x*np.arctan(σ_m*x/_σ)*ρ_st(x,θ,0,α,_σ,σ_m), -np.inf, np.inf)[0]

def σ_c(θ,α,σ_m):
    f = lambda _σ: R_prime_zero(θ,α,_σ,σ_m) - 1
    return optimize.root_scalar(f,bracket=[.5,2], x0=.5)
"""
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

from scipy.misc import derivative
def R_prime_zero(D,τ,θ):
    return derivative(lambda m: R(m,D,τ,θ), 0)

def σ_c(τ,θ):
    f = lambda D: R_prime_zero(D,τ,θ) - 1
    return optimize.root_scalar(f, x0=2)

if __name__=='__main__':
    θ = 4
    α = 1
    σ_m = 0
    τ = 1
    D = 2.95**2

    ms = np.linspace(-2,2,100)
    Rs = [R(m,D,τ,θ) - m for m in ms]
    plt.plot(ms,Rs)
    plt.plot(ms, 0*ms)
    plt.show()