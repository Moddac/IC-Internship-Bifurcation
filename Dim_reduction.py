# %%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
Please see the paper to understand all the computations and parameteres
By trying to solve the equation on the cumulants k_n  
"""
# Imports
import numpy as np
from math import factorial
from scipy.optimize import root
from scipy.integrate import solve_ivp
import time
import json
import os
import pickle
from scipy.stats import norm
from FileHandler_DimReduction import checkFile

# Parsing arguments
import argparse
parser = argparse.ArgumentParser(description="Solve the IVP with dimension reduction for moments or cumulant truncation scheme and store the results in a .json file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument("--scheme", required=True, help="scheme used for solving", choices=["cumulant", "moment"])
parser.add_argument("--alpha", required=True, help="alpha parameter", type=float)
parser.add_argument("--theta", required=True, help="theta paramter", type=float)
parser.add_argument("--sigma_m", required=True, help="sigma_m parameter", type=float)
parser.add_argument("--sigma_start", required=True, help="Starting point of sigmas", type=float)
parser.add_argument("--sigma_end", required=True, help="Ending point of sigmas", type=float)
group.add_argument("--N_sigma", help="Number of sigmas between sigma_start and simga_end", type=int)
parser.add_argument("--N", help="N parameters. Write each one wanted", nargs="*", type=int, metavar="N")
group.add_argument("--space_sigma", help="Space between 2 sigma points", type=float)
parser.add_argument("--fail_limit", help="Number of consecutive fails allowed before skipping a value of N", default=3, type=int)
parser.add_argument("--path", help="Path of the directory to save data", default="./Data")
parser.add_argument("-n", "--name", help="Name of the json file. Parameters are replaced by their values in default name.", default="Data_scheme_alpha_theta_sigma_m.json")
args = parser.parse_args()

# Explicit methods look really slow so skip it
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


def f_cumulant(t, k, α, θ, σ_m, σ):
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
    N = k.shape[0]
    _k = np.zeros(N+3)
    _k[1:N+1] = k  # N.B: _k[0] is never used, it is only to start at n=1

    # Starts at n=1, so we will keep it that way
    # First and second temrs are defined because of the kronecker symbol in the equations
    F = np.zeros(N)
    F[0] = θ*_k[1]
    F[1] = σ**2

    # -----Definition of f-----
    """
    To define f, a sum from 1 to n-1 which contains the three sums will be computed
    Then the last terms of the other sums are added
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
    k = solve_ivp(f_cumulant, (t0, t_end), k0,
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
    def g(x): return f_cumulant(0, x, α, θ, σ_m, σ)
    k = root(g, k0)

    return k


def f_moment(t, M, α, θ, σ_m, σ):
    """
    This is the function to solve dM/dt = f(M,t) 
    with M being a vector of function at a fix time t: M(t) = (M_1(t);...;M_N(t)) 
    """
    # -----Init-----
    N = M.shape[0]
    _M = np.zeros(N+4)
    _M[0] = 1
    _M[1:N+1] = M  # N.B: _M[0] is never used, it is only to start at n=1

    # Boundary conditions
    BC_M1 = 0
    BC_M2 = 0

    # Loading bell functions
    f1 = open('./functions/formula_mn1_'+str(N), "rb")
    f2 = open('./functions/formula_mn2_'+str(N), "rb")
    mn1 = pickle.load(f1)
    mn2 = pickle.load(f2)
    f1.close()
    f2.close()
    BC_M1 = mn1(np.reshape(M, (1, N)))
    BC_M2 = mn2(np.reshape(M, (1, N)))

    # ######## WORKING BUT REALLY LONG
    # Explicit formula
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
    Initial condition is a gaussian
    """
    # -----Init-----
    M0 = [norm.moment(n, loc=mean, scale=std) for n in range(1, N+1)]

    # -----Solver-----
    T = int(time.time()) - 1

    def stop_time(t, y, α, θ, σ_m, σ):
        τ = (int(time.time()) - T)
        return int(τ < 60)

    stop_time.terminal = True
    # stop_time.direction = +1

    M = solve_ivp(f_moment, (t0, t_end), M0,
                  args=(α, θ, σ_m, σ), method=method, events=stop_time)

    return M


def SolveMoment_Stationnary(N, α, θ, σ_m, σ):
    """
    Solving the cumulant ODE for 1 set of parameters
    And in stationnary state
    """
    # -----Init-----
    # N.B. HArd to find the right initial condition
    M0 = .5*np.ones(N)
    M0[5:7] = 1.5
    M0[7:N] = [4*i for i in range(1, N-6)]

    # -----Solver-----

    def g(x): return f_moment(0, x, α, θ, σ_m, σ)
    M = root(g, M0)

    return M
    # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
    # M1s.append(M1.x[0])


# %%
if __name__ == '__main__':
    """
    This script is to solve the differential equations of the cumulants k_n
    for a dimension reduction of N.

    Please note: for the data writing of the points, the values are automatically
    sorted by decreasing order. It is not inconvenient as the function is supposed to be 
    deacreasing according to sigma, but one has to remember it. 
    """

    # -----Init-----
    # Parameters
    SCHEME = args.scheme
    FAIL_LIMIT = args.fail_limit
    if args.name == "Data_scheme_alpha_theta_sigma_m.json":
        FILE_NAME = f"Data_{SCHEME}_{args.alpha}_{args.theta}_{args.sigma_m}.json"
    else:
        FILE_NAME = args.name 
    FILE_PATH = f"{args.path}/{FILE_NAME}"
    α = args.alpha # 1
    θ = args.theta # 4
    σ_m = args.sigma_m # .8
    N_σ = args.N_sigma
    space_σ = args.space_sigma
    if args.N:
        Ns = args.N
    else:
        Ns = []
    σ_start = args.sigma_start
    σ_end = args.sigma_end 

    if args.N_sigma:
        σs = np.linspace(σ_start, σ_end, N_σ)
    elif args.space_sigma:
        σs = np.arange(σ_start, σ_end, space_σ)

    solver = SolveCumulant_ODE if SCHEME == "cumulant" else SolveMoment_ODE

    # Creating or updating file
    indx_matrix = checkFile(FILE_PATH,
                            METHODS_IVP,
                            Ns,
                            σs,
                            α,
                            θ,
                            σ_m)
    
    # Loading data from created or updated file
    with open(FILE_PATH, "r") as file:
        data = json.load(file)

    t0 = 0
    t_end = 10e4



    # -----Solving-----
    # New parameters
    param = data["parameters"]
    methods, Ns, σs = param["methods"], param["Ns"], param["sigmas"]
    σs = np.array(σs)

    print("##########")
    print("Parameters: ")
    print(f"Ns={Ns}")
    print(f"First σ: {σs[0]}, Last σ: {σs[-1]}")
    print("##########")

    for method in methods:

        print(f"Solving with method: {method}")
        for N in Ns:

            # Init
            t1 = int(time.time())
            print(f"Solving for N={N}...")
            data_method = data[method][f"{N}"]
            σs_N = σs[indx_matrix[method][f"{N}"]]
            FAIL_COUNT = 0

            for i, σ in enumerate(σs_N):
                
                indx = indx_matrix[method][f"{N}"][i]
                if FAIL_COUNT < FAIL_LIMIT:
                    M1 = solver(N, t0, t_end, α, θ, σ_m, σ, method)
                    # Using the status to see if event stopped solving
                    success = (M1.status == 0)
                    data_method["success"].insert(indx, success)

                    if success:
                        data_method["points"].insert(indx, M1.y[0, -1])
                        FAIL_COUNT = 0
                    else:
                        data_method["points"].insert(indx, 0)
                        print(f"N={N}, σ={σ}, method={method} failed")
                        FAIL_COUNT += 1

                else:
                    data_method["success"].insert(indx, False)
                    data_method["points"].insert(indx, 0)
                    if FAIL_COUNT == FAIL_LIMIT:
                        print(
                            f"Too much failure for N={N}, method={method} so skiping the value {N}")
                        FAIL_COUNT += 1

            # Time
            t2 = int(time.time())
            data_method["time"] += t2-t1

            # Writing data
            with open(FILE_PATH, "w") as file:
                json.dump(data, file, indent=4)
# %%
