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
τ = 1
D = 1
θ = 1

def V(x):
    return x**4/4 - x**2/2

def dV(x):
    return x**3 - x

def d2V(x):
    return 3*x**2 - 1

def Φ(X):
    return np.sum(V(X)) + θ*np.sum((X[:,None]-X[:,None])**2)/(2*X.size)

def gradΦ(X):
    N = X.size
    G = np.zeros(N)

    for i in range(N):
        G[i] = dV(X[i]) + θ*np.sum(X[i]-X)/N

    return G 

def H(X):
    N = X.size
    Hess = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                Hess[i,j] = -θ/N
            else:
                Hess[i,j] = d2V(X[i]) + θ
    return Hess

def f_m(X):
    N = X.size
    return -1/D*(Φ(X) + τ/2*np.linalg.norm(gradΦ(X))**2) * np.abs(np.linalg.det(np.eye(N,N) + τ*H(X)))                        

def ρ_st(x):
    if f'{θ}' not in Zs.keys():
        Zs[f'{θ}'] = integrate.quad(lambda y: np.exp(-f_m(y)),-np.inf,np.inf)[0]
    Z = Zs[f'{θ}']
    return np.exp(-f_m(x))/Z

if __name__=='__main__':
    print(ρ_st([0]))