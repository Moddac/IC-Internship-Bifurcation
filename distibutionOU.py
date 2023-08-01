import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from findiff import FinDiff

Zs = {}  # Dict used for keeping the integration constant in the memory

def V(x):
    return x**4/4 - x**2/2


def dV(x):
    return x**3 - x


def d2V(x):
    return 3*x**2 - 1


def Φ(x, m, θ):
    return V(x) + .5*θ*(x-m)**2


def gradΦ(x, m, θ):
    return dV(x) + θ*(x-m)


def H(x, θ):
    return d2V(x) + θ


def f_m(x, m, D, τ, θ):
    return 1/D*(Φ(x, m, θ) + τ/2*gradΦ(x, m, θ)**2)


def ρ_st(x, m, D, τ, θ):
    if f'{m},{D},{τ},{θ}' not in Zs.keys():
        Zs[f'{m},{D},{τ},{θ}'] = integrate.quad(
            lambda y: np.exp(-f_m(y, m, D, τ, θ))*np.abs(1 + τ*H(y, θ)), -np.inf, np.inf)[0]
    Z = Zs[f'{m},{D},{τ},{θ}']
    return np.exp(-f_m(x, m, D, τ, θ))*np.abs(1 + τ*H(x, θ))/Z


def R(m, σ, τ, θ):
    D = σ**2 / 2
    return integrate.quad(lambda x: x*ρ_st(x, m, D, τ, θ), -np.inf, np.inf)[0]  


def plotSelfConsistency(τ, θ, α, σ_start, σ_end):
    """
    Plots values of R(m) - m and R'(m) - 1
    Can be used to have boundaries around sigma_c and use it to have boundaries for root_scalar
    """
    # -----Init-----
    σs = np.linspace(σ_start, σ_end, 50)
    ms = np.linspace(-.2, .2, 50)
    fig, ax = plt.subplots()
    d_dm = FinDiff(0, ms[1]-ms[0])

    # -----Plot-----
    for σ in σs:
        D = σ**2
        Rs = [R(m, D, τ, θ) - m for m in ms]
        dRs_dm = d_dm(np.array([R(m, D, τ, θ)-m for m in ms]))
        
        ax.set_xlabel("m")
        ax.plot(ms, 0*ms)
        ax.plot(ms, Rs, label="R(m)-m")
        ax.plot(ms, dRs_dm, label="R'(m)-1")
        ax.set_title("σ=%.3f" % σ)
        ax.legend()
        plt.pause(0.01)
        ax.clear()


def σ_c(τ, θ):
    """
    finding the critical value sigma_c with equation 
    R'(0) = 1
    sigma_before and sigma_after are respectively values of sigma beofre and after the critical value
    """
    # -----Optimize-----
    h = 1e-6
    def f(σ): return (R(h, σ, τ, θ) - h - R(0, σ, τ, θ)) / h

    root = optimize.root_scalar(f, bracket=[0.1, 100.0])

    return root.root