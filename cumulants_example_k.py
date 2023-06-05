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
import os

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

    param = data["parameters"]
    methods, Ns, σs = param["methods"], param["Ns"], param["sigmas"]
    σs = np.array(σs)

    for method in methods:
        for N in Ns:

            info = data[method][f"{N}"]
            M1s, time, success = info["points"], info["time"], info["success"]
            M1s = np.array(M1s)

            plt.scatter(σs[success], M1s[success],
                        label=f"N={N}, time={time}s")

        plt.xlabel("σ")
        plt.ylabel("m")
        plt.legend()
        plt.title(f"Mean with cumulant solving ODE with method '{method}'")
        plt.savefig(f"Figs/{method}.png")
        plt.show()
        plt.close()


def checkFile(file_name, methods, Ns, σs, α, θ, σ_m):
    """
    This checks if the data file already exists.
    It will search locally for a Data/data_parameters.json file and create
    folders and file if they don't exist

    If file already exists, checks which methods, N and sigmas have been done
    and update the file with new parameters

    Also returns a list of sigma to be computed, according to those which have already been done
    and the new sigmas
    """
    if not os.path.exists("./Data"):
        os.mkdir("./Data")

    file_path = f"./Data/{file_name}"
    if not os.path.isfile(file_path):
        # If there is no file, create it with the parameters
        # Writing parameters
        data = {
            "parameters": {}
        }
        for method in methods:
            data[f"{method}"] = {
                N: {
                    "points": [],
                    "time": 0,
                    "success": []
                }
                for N in Ns}
        data["parameters"] = {
            "methods": methods,
            "Ns": Ns,
            "sigmas": list(),
            "alpha": α,
            "theta": θ,
            "sigma_m": σ_m,
        }
        json_dic = json.dumps(data, indent=4)
        with open(file_path, "w") as file:
            file.write(json_dic)

        σs_matrix = {}
        for method in methods:
            σs_matrix[f"{method}"] = {}
            for N in Ns:
                σs_matrix[f"{N}"] = σs

        return methods, Ns, σs, σs_matrix

    else:
        # If the file exists, check the Ns, sigmas and methods
        with open(file_path, "r") as file:
            data = json.load(file)

        param = data["parameters"]
        _methods, _Ns, _σs = param["methods"], param["Ns"], param["sigmas"]

        σs_union = np.union1d(σs, _σs).tolist()
        # Values in sigmas but not in _sigmas
        new_σs = np.setdiff1d(σs, _σs).tolist()
        Ns_union = np.union1d(Ns, _Ns).tolist()
        print(Ns, _Ns, Ns_union)
        methods_union = np.union1d(methods, _methods).tolist()
        σs_matrix = {}

        # Updating file with new parameters
        param["Ns"] = Ns_union
        param["methods"] = methods_union
        param["sigmas"] = σs_union

        """
        For now: data points for a model are writen only if all the sigmas are done
        So we can check either if the lis "points" is empty or not
        If empty: all the sigmas must be done
        If not empty: only the new sigmas
        TODO: write on the file sigma per sigma?
        """
        for method in methods_union:
            σs_matrix[f"{method}"] = {}
            σs_matrix_method = σs_matrix[f"{method}"]

            if method not in _methods:
                data[f"{method}"] = {}

            data_method = data[f"{method}"]

            for N in Ns_union:
                str_N = str(N)

                if str_N not in data_method.keys():
                    data_method[str_N] = {
                        "points": [],
                        "time": 0,
                        "success": []
                    }

                if str_N not in data_method.keys() or len(data_method[str_N]["points"]) == 0:
                    # Case where N was not defined or was not done
                    σs_matrix_method[str_N] = σs_union

                else:
                    # Else, only take the new sigmas
                    σs_matrix_method[str_N] = new_σs

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return methods_union, Ns_union, σs_union, σs_matrix


# %%
if __name__ == '__main__':

    PLOT = False
    SOLVE = True

    if SOLVE:
        # -----Init-----
        Ns = [4, 8]  # , 16, 24]
        α = 1
        θ = 4
        σ_m = .8
        N_σ = 30
        σ_start, σ_end = 1.8, 2.
        σs = np.linspace(1.8, 2., N_σ)
        FILE_NAME = f"Data.json"
        methods, Ns, σs, σs_matrix = checkFile(FILE_NAME,
                                               METHODS_IVP,
                                               Ns,
                                               σs,
                                               α,
                                               θ,
                                               σ_m)
        print(methods, Ns, σs_matrix)
        with open(f"./Data/{FILE_NAME}", "r") as file:
            data = json.load(file)

        t0 = 0
        t_end = 10e4

        # -----Solving-----
        print("##########")
        print("Parameters: ")
        print(f"Ns={Ns}")
        print(f"First σ: {σs[0]}, Last σ: {σs[-1]}, N_σ: {N_σ}")
        print("##########")

        for method in methods:

            print(f"Solving with method: {method}")
            for N in Ns:

                # Init
                t1 = int(time.time())
                print(f"Solving for N={N}...")
                _data = data[method][f"{N}"]
                σs = σs_matrix[f"{method}"][f"{N}"]

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
        file_path = "Data.json"
        plotData(file_path)

# %%
