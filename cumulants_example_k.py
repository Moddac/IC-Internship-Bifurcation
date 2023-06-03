# %%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
By trying to solve the equation on the cumulants k_n  
"""

import numpy as np
from math import factorial
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import json

# Explicit methods look really slow
METHODS_IVP = [  # "RK45",
    # "RK23",
    # "DOP853",
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


def f(t, k, α, θ, σ_m, σ):
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
    N = k.shape[0]
    _k = np.zeros(N+3)
    _k[1:N+1] = k  # N.B: _k[0] is never used

    # Starts at n=1, so we will keep it that way
    F = np.zeros(N)
    F[0] = θ*_k[1]
    F[1] = σ**2  # it is not sigma² / 2 because we multiply by n=2

    # -----Definition of f-----
    """
    To define f, a sum from 1 to n-1 which contains the three sums will be computed
    Then the last terms of the sum are added
    And we suppose B.C.: k_{N+1} = k_{N+2} = 0
    """
    for n in range(1, N+1):
        Σ = np.sum([
            σ_m**2/2*_k[i]*_k[n-i] / (factorial(i-1)*factorial(n-i-1))
            - 3*_k[i]*_k[n-i+2] / (factorial(i-1)*factorial(n-i))
            - np.sum([
                _k[i]*_k[j]*_k[n+2-i-j] /
                (factorial(i-1)*factorial(j-1)*factorial(n-i-j+1))
                for j in range(1, n-i+2)])
            for i in range(1, n)])

        F[n-1] += n*(
            (
                α-θ+σ_m**2*(ν+(n-1)/2))*_k[n]
            - _k[n+2]
            + factorial(n-1)*(
                Σ
                - 3*_k[n]*_k[2]/factorial(n-1)
                - _k[n]*_k[1]**2/factorial(n-1)
            )
        )

    return F


def SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, method):
    """
    Solving the cumulant ODE for 1 set of parameters
    """
    # -----Init-----
    k0 = np.zeros(N)
    k0[0] = 1
    k0[1] = 1.5**2

    # -----Solver-----
    k = solve_ivp(f, (t0, t_end), k0,
                  args=(α, θ, σ_m, σ), method=method)

    return k


def SolveCumulant_Stationnary(N, α, θ, σ_m, σ):
    """
    Solving the cumulant ODE for 1 set of parameters
    And in stationnary state
    """
    # -----Init-----
    k0 = 0.01*np.ones(N)
    k0[0] = .5
    k0[1] = .5

    # -----Solver-----
    def g(x): return f(0, x, α, θ, σ_m, σ)
    k = root(g, k0)

    return k

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
        plt.title(f"Mean with cumulant solving ODE with method '{method}'")
        plt.savefig(f"Figs/{method}.png")
        plt.show()
        plt.close()

# %%
if __name__ == '__main__':

    PLOT = False
    SOLVE = True
    FILE_NAME = "Data_k.json"

    if SOLVE:
        # -----Init-----
        Ns = [4, 8]#, 16, 24]
        α = 1
        θ = 4
        σ_m = .8
        N_σ = 150
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

                    M1 = SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, method)
                    _data["success"].append(M1.success)

                    if M1.success:
                        _data["points"].append(M1.y[0, -1])
                    else:
                        _data["points"].append(0)
                        print(f"N={N}, σ={σ}, method={method} failed")

                    # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
                    # M1s.append(M1.x[0])

                # Time
                t2 = int(time.time())
                _data["time"] = t2-t1

                # Writing data
                json_data = json.dumps(data, indent=4)
                with open(FILE_NAME, "w") as file:
                    file.write(json_data)

    if PLOT:
        file_path = "Data_01_06_k.json"
        plotData(file_path)

# %%
