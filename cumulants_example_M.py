"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
By trying to solve the equation on the moments M_n  
"""
# %%
import numpy as np
from numpy.random import normal
# from sympy import bell
from math import factorial, comb
from scipy.optimize import fsolve, newton, root
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
import time
import json

METHODS_IVP = [#"RK45", 
               #"RK23",
               #"DOP853",
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
    _M = np.zeros(N+4)
    _M[0] = 1
    _M[1:N+1] = M
    # Boundary conditions
    BC_M1 = 0
    BC_M2 = 0

    f1 = open('./functions/formula_mn1_'+str(N), "rb")
    f2 = open('./functions/formula_mn2_'+str(N), "rb")
    mn1 = pickle.load(f1)
    mn2 = pickle.load(f2)
    f1.close(); f2.close()
    BC_M1 = mn1(np.reshape(M, (1, N)))
    BC_M2 = mn2(np.reshape(M, (1, N)))

    # ######## WORKING BUT REALLY LONG
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


def SolveMoment_ODE(N, t0, t_end, α, θ, σ_m, σ, method, mean=1, std=1):
    """
    Solving the cumulant ODE for 1 set of parameters
    """
    # -----Init-----
    M0 = [norm.moment(n, loc=mean, scale=std) for n in range(1,N+1)]

    # -----Solver-----
    M = solve_ivp(f, (t0, t_end), M0,
                  args=(α, θ, σ_m, σ), method=method)

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

def plotData(file_path):
    """
    File must be the same format as Data.json created
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        
    methods, Ns, σs = data["parameters"].values()
    σs = np.array(σs)

    for method in methods:
        for N in Ns:

            M1s, time, success =  data[method][f"{N}"].values()
            M1s = np.array(M1s)

            plt.scatter(σs[success], M1s[success], label=f"N={N}, time={time}s")

        plt.xlabel("σ")
        plt.ylabel("m")
        plt.legend()
        plt.title(f"Mean with moments solving ODE with method '{method}'")
        plt.show()
        plt.savefig(f"Figs/{method}.png")
        plt.close()


# %%
if __name__ == '__main__':

    PLOT = False
    SOLVE = True
    FILE_NAME = f"Data_M_{int(time.time())}.json"

    # -----Init-----
    Ns = [4, 8, 16, 24]
    α = 1
    θ = 4
    σ_m = .8
    N_σ = 2
    σs = np.linspace(1.8, 2., N_σ)

    # Writing parameters
    data = {
        "parameters": {}
    }
    for method in METHODS_IVP:
        data[f"{method}"] = {
            N: {
                "points": [],
                "time": 0,
                "success": []
            }
            for N in Ns}
    data["parameters"] = {
        "methods": METHODS_IVP,
        "Ns": Ns,
        "sigmas": list(σs)
    }
    json_dic = json.dumps(data, indent=4)
    with open(FILE_NAME, "w") as file:
        file.write(json_dic)

    t0 = 0
    t_end = 10e4

    # -----Solving-----
    print("##########")
    print("Parameters: ")
    print(f"Ns={Ns}")
    print(f"First σ: {σs[0]}, Last σ: {σs[-1]}, N_σ: {N_σ}")
    print("##########")

    for method in METHODS_IVP:

        print(f"Solving with method: {method}")
        for N in Ns:

            # Init
            t1 = int(time.time())
            print(f"Solving for N={N}...")
            _data = data[method][N]

            for i, σ in enumerate(σs):

                print(σ)
                M1 = SolveMoment_ODE(N, t0, t_end, α, θ, σ_m, σ, method)
                _data["success"].append(M1.success)

                if M1.success:
                    _data["points"].append(M1.y[0, -1])
                else:
                    _data["points"].append(0)
                    print(f"N={N}, σ={σ}, method={method} failed")

                # M1 = SolveMoment_Stationnary(N, α, θ, σ_m, σ)
                # M1s.append(M1.x[0])

            # Time
            t2 = int(time.time())
            _data["time"] = t2-t1

            # Writing data
            json_data = json.dumps(data, indent=4)
            with open(FILE_NAME, "w") as file:
                file.write(json_data)

    if PLOT:
        file_path = "Data_M.json"
        plotData(file_path)
