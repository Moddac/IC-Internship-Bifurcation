# %%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
Please see the paper to understand all the computations and parameteres
By trying to solve the equation on the cumulants k_n  
"""
# Imports
import numpy as np
from math import factorial, comb
from scipy.optimize import root
from scipy.integrate import solve_ivp
import time
import json
import os
import pickle
from scipy.stats import norm
from FileHandler_DimReduction import checkFile
from tools import Isserlis

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
parser.add_argument("--epsilon", help="Epsilon parameter for OU noise", default=1., type=float)
parser.add_argument("--fail_limit", help="Number of consecutive fails allowed before skipping a value of N", default=3, type=int)
parser.add_argument("--path", help="Path of the directory to save data", default="./Data_DimReduction")
parser.add_argument("-n", "--name", help="Name of the json file. Parameters are replaced by their values in default name.", default="Data_scheme_alpha_theta_sigma_m.json")
parser.add_argument("--delete_file", help="Delete data file if the name are the same. Warning: turning this option to true will delete the data file without checking anything", action="store_true")
args = parser.parse_args()

# Explicit methods look really slow so skip it
METHODS_IVP = [  # "RK45",
    # "RK23",
    # "DOP853",
    # "Radau",
    "BDF"]
    # "LSODA"]

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
    _M = np.zeros(N+4) # _M represents the vector (M_0, M_1, ..., M_n, M_{n+1}, M_{n+2}, M_{-1}) with B.C on M_0, M_{n+1}, M_{n+2}, M_{-1}
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

def f_moment_OU(t, u, α, θ, σ, ε):
    """
    This is the function to solve dM/dt = f(M,t) for a OU noise 
    See paper for equation
    """
    # -----Init-----
    N = u.shape[0] - 1
    M = np.zeros((N+3, 2))
    if t==0:
        E_ηt = 0
        E_ηtηt = 0
    else:
        E_ηt = norm.moment(1, loc=0, scale=np.sqrt(1-np.exp(-2*t)))
        E_ηtηt = norm.moment(2, loc=0, scale=np.sqrt(1-np.exp(-2*t)))
    M[0,0] = 1
    M[1,1] = u[N] # X_t * η_t
    M[0,1] = E_ηt
    M[1:N+1, 0] = u[:N]
    for n in range(2, N+1):
        M[n,1] = np.sum([
            comb(n,k)*M[1,0]**(n-k)*Isserlis(k, (M[2,0]-M[1,0]**2), (M[1,1]-M[1,0]*M[0,1]))
        for k in range(n+1)])

    # Loading bell functions
    fN1 = open('./functions/formula_mn1_'+str(N), "rb")
    fN2 = open('./functions/formula_mn2_'+str(N), "rb")
    mn1 = pickle.load(fN1)
    mn2 = pickle.load(fN2)
    fN1.close(); fN2.close()
    for p in range(1):
        M[N+1, p] = mn1(np.reshape(M[1:N+1, p], (1, N)))
        M[N+2, p] = mn2(np.reshape(M[1:N+1, p], (1, N)))

    F = np.zeros(N+1)

    # -----Definition of f-----
    """
    We suppose B.C. on M_{N+1} and M_{N+2} to be the ones in the paper 
    """
    for i in range(1, N+1):
        F[i-1] = i*(
            - M[i+2,0] \
            - M[i,0]*(θ - α) \
            + θ*M[1,0]*M[i-1,0] \
            + ζ*σ*M[i-1,1] / ε
        )
    F[N] = (
            - M[3,1] \
            - M[1,1]*(θ - α) \
            + θ*M[1,0]*M[0,1] \
            + ζ*σ*E_ηtηt / ε \
            - M[1,1] / ε**2
        )
        
    return F


def SolveMoment_ODE(N, t0, t_end, α, θ, σ_m, σ, ε, method, mean=1, std=1, noise="white"):
    """
    Solving the cumulant ODE for 1 set of parameters
    Initial condition is a gaussian
    """
    if noise=="white":
        # -----Init-----
        M0 = [norm.moment(n, loc=mean, scale=std) for n in range(1, N+1)]

        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, α, θ, σ_m, σ):
            τ = (int(time.time()) - T)
            return int(τ < 300)

        stop_time.terminal = True
        # stop_time.direction = +1

        M = solve_ivp(f_moment, (t0, t_end), M0,
                    args=(α, θ, σ_m, σ), method=method, events=stop_time)

        return M
    
    if noise=="OU":
        # -----Init-----
        u0 = [norm.moment(n, loc=mean, scale=std) for n in range(1, N+1)]
        u0 += [0] # X_0 * η_0
        u0 = np.array(u0)
            
        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, α, θ, σ, ε):
            τ = (int(time.time()) - T)
            return int(τ < 300)
        stop_time.terminal = True

        u = solve_ivp(f_moment_OU, (t0, t_end), u0,
                    args=(α, θ, σ, ε), method=method, events=stop_time)
        print(u.y[:, -1])

        return u


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

    # Deleting file if asked
    if args.delete_file and os.path.isfile(FILE_PATH):
        os.remove(FILE_PATH)

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
    ε = args.epsilon
    ζ = 1 / np.sqrt(2)

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
    t_end = 10e6

    ####################################################################################
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

                    M1 = solver(N, t0, t_end, α, θ, σ_m, σ, ε, method, noise="OU")
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
