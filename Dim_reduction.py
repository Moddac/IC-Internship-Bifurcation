# %%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
Please see the paper to understand all the computations and parameteres
By trying to solve the equation on the cumulants k_n  
"""
# Imports
import numpy as np
import time
import json
import os
from FileHandler_DimReduction import checkFile

from functionCumulant import SolveCumulant_ODE
from functionsMoment import SolveMoment_ODE

# Parsing arguments
import argparse
parser = argparse.ArgumentParser(description="Solve the IVP with dimension reduction for moments or cumulant truncation scheme and store the results in a .json file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument("--scheme", required=True,
                    help="scheme used for solving", choices=["cumulant", "moment"])
parser.add_argument("--noise", required=True, help="Type of noise", choices=["white", "OU"])
parser.add_argument("--alpha", required=True,
                    help="alpha parameter", type=float)
parser.add_argument("--theta", required=True,
                    help="theta paramter", type=float)
parser.add_argument("--sigma_m", required=True,
                    help="sigma_m parameter", type=float)
parser.add_argument(
    "--epsilon", help="Epsilon parameter for OU noise", default=1., type=float)
parser.add_argument("--gamma", help="Parameter for inertia. 0 means inertia is not considered", default=0., type=float)
parser.add_argument("--sigma_start", required=True,
                    help="Starting point of sigmas", type=float)
parser.add_argument("--sigma_end", required=True,
                    help="Ending point of sigmas", type=float)
group.add_argument(
    "--N_sigma", help="Number of sigmas between sigma_start and simga_end", type=int)
parser.add_argument("--N", help="N parameters. Write each one wanted",
                    nargs="*", type=int, metavar="N")
group.add_argument(
    "--space_sigma", help="Space between 2 sigma points", type=float)
parser.add_argument(
    "--fail_limit", help="Number of consecutive fails allowed before skipping a value of N", default=3, type=int)
parser.add_argument(
    "--path", help="Path of the directory to save data", default="./Data_DimReduction")
parser.add_argument("--name", help="Name of the json file. Parameters are replaced by their values in default name.",
                    default="Data_scheme_noise_alpha_theta_sigma_m_gamma_epsilon.json")
parser.add_argument("--delete_file", help="Delete data file if the name are the same. Warning: turning this option to true will delete the data file without checking anything, use at your own risk.", action="store_true")
parser.add_argument("--stop_time", help="Time to stop the solver and consider a failure", type=float, default=300)
args = parser.parse_args()


# HYPER PARAMETERS
# Explicit methods look really slow so skip it
METHODS_IVP = [
    # "RK45",
    # "RK23",
    # "DOP853",
    # "Radau",
    # "BDF",
    "LSODA"
    ]
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
P = 4
K = 2


# %%
if __name__ == '__main__':
    """
    This script is to solve the differential equations of the cumulants k_n
    for a dimension reduction of N.

    Please note: for the data writing of the points, the values are automatically
    sorted by decreasing order. It is not inconvenient as the function is supposed to be 
    deacreasing according to sigma, but one has to remember it. 
    """

    ###############################################################################
    # -----Init-----
    # Parameters
    SCHEME = args.scheme
    FAIL_LIMIT = args.fail_limit
    STOP_TIME = args.stop_time
    if args.name == "Data_scheme_noise_alpha_theta_sigma_m_gamma_epsilon.json":
        if args.noise=="white":
            FILE_NAME = f"Data___{SCHEME}___{args.noise}___alph{args.alpha:_}___thet{args.theta:_}___sigm_m{args.sigma_m:_}___gam{args.gamma:_}.json"
        if args.noise=="OU" and args.gamma==0.:
            FILE_NAME = f"Data___{SCHEME}___{args.noise}___alph{args.alpha:_}___thet{args.theta:_}___sigm_m{args.sigma_m:_}___gam{args.gamma:_}___eps{args.epsilon:_}___P{P}.json"
        if args.noise=="OU" and args.gamma!=0.:
            FILE_NAME = f"Data___{SCHEME}___{args.noise}___alph{args.alpha:_}___thet{args.theta:_}___sigm_m{args.sigma_m:_}___gam{args.gamma:_}___eps{args.epsilon:_}___P{P}___K{K}.json"
    else:
        FILE_NAME = args.name
    FILE_PATH = f"{args.path}/{FILE_NAME}"

    # Deleting file if asked
    if args.delete_file and os.path.isfile(FILE_PATH):
        os.remove(FILE_PATH)

    noise = args.noise
    if noise=="OU":
        ζ = 1 / np.sqrt(2)

    α = args.alpha  
    θ = args.theta  
    σ_m = args.sigma_m  
    ε = args.epsilon
    γ = args.gamma
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
                            σ_m,
                            γ,
                            ε)

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

                    M1, success = solver(N, P, K, t0, t_end, α, θ, σ_m,
                                        σ, γ, ε, method, noise, STOP_TIME) 
                    # Using the status to see if event stopped solving
                    data_method["success"].insert(indx, success)

                    if success:
                        data_method["points"].insert(indx, M1)
                        FAIL_COUNT = 0
                    else:
                        data_method["points"].insert(indx, 0)
                        print(f"N={N}, σ={σ}, method={method} failed")
                        print("Value:", M1)
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
            if FAIL_COUNT > FAIL_LIMIT:
                data_method["time"] = f"FAILED ({data_method['time']}s)"


            # Writing data
            with open(FILE_PATH, "w") as file:
                json.dump(data, file, indent=4)
# %%
