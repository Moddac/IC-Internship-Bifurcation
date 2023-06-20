"""
Find the critical value of sigma  for an IPS with sigma(x) = sqrt(sigma**2 + (sigma_m*x)**2)
Case sigma_m = 0 is white noise
"""
from findiff import FinDiff
import numpy as np
from numpy.random import normal
from scipy import integrate
import scipy.optimize as optimize
from numpy.random import uniform
import matplotlib.pyplot as plt

Zs = {}  # Dict used for keeping the integration constant in the memory

def σ(x, _σ, σ_m):
    return np.sqrt(_σ**2 + (σ_m*x)**2)


def dσ(x, _σ, σ_m):
    return (σ_m**2*x)/np.sqrt(_σ**2 + (σ_m*x)**2)


def V_α(x, α):
    return x**4/4 - α*x**2/2


def F_α(x, α):
    return α*x - x**3


def F_hat_α(x, α, _σ, σ_m):
    return F_α(x, α) + σ(x, _σ, σ_m)*dσ(x, _σ, σ_m)/2


def f_m(x, m, α, θ, _σ, σ_m):
    if σ_m == 0:
        return f_m_det(x, m, α, _σ, σ_m)
    return 2*integrate.quad(lambda y: (-F_hat_α(y, α, _σ, σ_m) + θ*(y-m))/(σ(y, _σ, σ_m)**2), 0, x)[0] + np.log(σ(x, _σ, σ_m)**2)


def f_m_det(x, m, α, _σ, σ_m):
    if σ_m == 0:
        return 2/(_σ**2)*(V_α(x, α) + θ/2*(x-m)**2)
    return -(α-θ-.5*σ_m**2+(_σ/σ_m)**2)/(σ_m**2)*np.log(1+(σ_m*x/_σ)**2) + (x/σ_m)**2 - 2*θ*m/(_σ*σ_m)*np.arctan(σ_m/_σ*x)


def ρ_st(x, θ, m, α, _σ, σ_m):
    if f'{θ},{m},{α},{_σ},{σ_m}' not in Zs.keys():
        Zs[f'{θ},{m},{α},{_σ},{σ_m}'] = integrate.quad(
            lambda y: np.exp(-f_m_det(y, m, α, _σ, σ_m)), -np.inf, np.inf)[0]
    Z = Zs[f'{θ},{m},{α},{_σ},{σ_m}']
    return np.exp(-f_m_det(x, m, α, _σ, σ_m))/Z


def R(θ, m, α, _σ, σ_m):
    return integrate.quad(lambda x: x*ρ_st(x, θ, m, α, _σ, σ_m), -np.inf, np.inf)[0]


def R_prime_zero(θ, α, _σ, σ_m):
    if σ_m == 0:
        return 2*θ/(_σ**2)*integrate.quad(lambda x: x**2*ρ_st(x, θ, 0, α, _σ, σ_m), -np.inf, np.inf)[0]
    else:
        return 2*θ/(_σ*σ_m)*integrate.quad(lambda x: x*np.arctan(σ_m*x/_σ)*ρ_st(x, θ, 0, α, _σ, σ_m), -np.inf, np.inf)[0]


def plotSelfConsistency(θ, α, σ_m, σ_start, σ_end):
    """
    Plots values of R(m) - m and R'(m) - 1
    Can be used to have boundaries around sigma_c and use it to have boundaries for root_scalar
    """
    # -----Init-----
    σs = np.linspace(σ_start, σ_end, 50)
    ms = np.linspace(-.5, .5, 50)
    fig, ax = plt.subplots()
    d_dm = FinDiff(0, ms[1]-ms[0])

    # -----Plot-----
    for _σ in σs:
        Rs = [R(θ, m, α, _σ, σ_m) - m for m in ms]
        dRs_dm = d_dm(np.array([R(θ, m, α, _σ, σ_m)-m for m in ms]))

        ax.set_xlabel("m")
        ax.plot(ms, 0*ms)
        ax.plot(ms, Rs, label="R(m)-m")
        ax.plot(ms, dRs_dm, label="R'(m)-1")
        ax.set_title("σ=%.3f" % _σ)
        ax.legend()
        plt.pause(0.01)
        ax.clear()


def σ_c(θ, α, σ_m, σ_before, σ_after):
    """
    finding the critical value sigma_c with equation 
    R'(0) = 1
    sigma_before and sigma_after are respectively values of sigma beofre and after the critical value
    """

    def f(_σ): return R_prime_zero(θ, α, _σ, σ_m) - 1
    return optimize.root_scalar(f, bracket=[σ_before, σ_after])


if __name__ == '__main__':
    θ = 4
    α = 1
    σ_m = 1.
    σ_start = 1.7
    σ_end = 2.2

    plotSelfConsistency(θ, α, σ_m, σ_start, σ_end)
    print(σ_c(θ, α, σ_m, σ_start, σ_end))
