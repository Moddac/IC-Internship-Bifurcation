"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
By trying to solve the equation on the moments M_n  
"""
# %%
import numpy as np
from numpy.random import normal
from sympy import bell
from math import factorial, comb
from scipy.optimize import fsolve, newton, root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import dill

METHODS_IVP = ["RK45", 
               "RK23",
               "DOP853",
               "Radau",
               "BDF",
               "LSODA"]
METHODS_ROOT = ["hybr",
                "lm",
                "broyden1",
                "broyden2",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
                "df-sane"]
ν = .5


def f(t, M, α, θ, σ_m, σ):
    """
    This is the function to solve dM/dt = f(M,t) 
    with M being a vector of function at a fix time t: M(t) = (M_1(t);...;M_N(t)) 
    """
    # -----Init-----
    N = M.shape[0]
    _M = np.ones(N+4)
    # _M[0] = 1
    _M[1:N+1] = M
    # Boundary conditions
    BC_M1 = 0
    BC_M2 = 0

    f1 = open('./Dim_reduction/code/functions/formula_mn1_'+str(N), "rb")
    f2 = open('./Dim_reduction/code/functions/formula_mn2_'+str(N), "rb")
    mn1 = dill.load(f1)
    mn2 = dill.load(f2)
    BC_M1 = mn1(np.reshape(M, (1, N)))
    BC_M2 = mn2(np.reshape(M, (1, N)))

    # BC_M1 = -np.sum([
    #     (-1)**(l-1)*factorial(l-1)*bell(N+1, l, M)
    #     for l in range(2, N+2)])

    # Σ1 = np.sum([
    #     comb(N+2, k)*_M[k]*_M[N+2-k]
    #     for k in range(2, N+1)])
    # Σ2 = np.sum([
    #     (-1)**(l-1)*factorial(l-1)*bell(N+2, l, M)
    #     for l in range(3, N+3)])
    # BC_M2 = (
    #     (N+2)*BC_M1*_M[1]
    #     + .5*Σ1
    #     - Σ2
    # )

    _M[N+1] = BC_M1
    _M[N+2] = BC_M2
    F = np.zeros(N)

    # -----Definition of f-----
    """
    We suppose B.C. on M_{N+1} and M_{N+2} to be the ones in the paper 
    """
    for n in range(1, N+1):
        F[n-1] = n*(
            (α - θ + n/2*σ_m**2)*_M[n]
            - _M[n+2]
            + (n-1)/2*σ**2*_M[n-2]
            + θ*_M[1]*_M[n-1]
        )
    return F


def SolveMoment_ODE(N, t0, t_end, α, θ, σ_m, σ, mean=1, std=1):
    """
    Solving the cumulant ODE for 1 set of parameters
    """
    # -----Init-----
    # M0 = np.zeros(N)
    # M0[0] = 1
    # M0[1] = .5
    from scipy.stats import norm
    M0 = [norm.moment(n, loc=mean, scale=std) for n in range(1,N+1)]

    # -----Solver-----
    M = solve_ivp(f, (t0, t_end), M0,
                  args=(α, θ, σ_m, σ), method='Radau')

    return M


def SolveMoment_Stationnary(N, α, θ, σ_m, σ):
    """
    Solving the cumulant ODE for 1 set of parameters
    And in stationnary state
    """
    # -----Init-----
    #N.B. HArd to find the right initial condition
    M0 = .5*np.ones(N)
    M0[5:7] = 1.5
    M0[7:N] = [4*i for i in range(1,N-6)]
    

    # -----Solver-----
    def g(x): return f(0, x, α, θ, σ_m, σ)
    M = root(g, M0)

    return M


# %%
if __name__ == '__main__':

    # -----Init-----
    Ns = [8, 10]
    α = 1
    θ = 4
    σ_m = .8
    N_σ = 100
    σs = np.linspace(1.8, 2., 200)

    t0 = 0
    t_end = 5e6

    # -----Solving-----
    print("##########")
    print("Parameters: ")
    print(f"Ns={Ns}")
    print(f"First σ: {σs[0]}, Last σ: {σs[-1]}")
    print("##########")
    print("Starting solving...")
    for N in Ns:
        print(f"Solving for N={N}...")
        M1s = []

        for i, σ in enumerate(σs):
            if i % 10 == 0:
                print(f"σ={σ}")
            # M1 = SolveMoment_ODE(N, t0, t_end, α, θ, σ_m, σ)
            # print(M1.y[:, -1])
            # M1s.append(M1.y[0, -1])

            M1 = SolveMoment_Stationnary(N, α, θ, σ_m, σ)
            M1s.append(M1.x[0])

        # -----Plot-----
        plt.scatter(σs, M1s, label=f"N={N}")

    plt.xlabel("σ")
    plt.ylabel("m")
    plt.legend()
    plt.title("Mean with moment method")
    plt.show()
