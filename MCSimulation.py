"""
Attempt to solve interacting particle SDE with Euler-Maruyama shceme for different noises
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import scipy.integrate as integrate
from FileHandler_MCMC import checkFile
import json
import os

# Parsing arguments
import argparse
parser = argparse.ArgumentParser(description="Solve the IVP with Monte-Carlo simulation and store the results in a .json file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N", required=True, help="Number of step for the simulation", type=int)
parser.add_argument("--N_p", required=True, help="Number of particles", type=int)
parser.add_argument("--dt", help="Time space between two steps", default=0.01, type=float)
parser.add_argument("--theta", required=True, help="Theta paramter", type=float)
parser.add_argument("--beta_start", required=True, help="Starting point of betas", type=float)
parser.add_argument("--beta_end", required=True, help="Ending point of betas", type=float)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--N_beta", help="Number of beta between beta_start and beta_end", type=int)
group.add_argument("--step_beta", help="Step size between 2 beta points", type=float)
parser.add_argument("--noise", help="Type of noise", default="white", choices=["white", "OU", "H"])
parser.add_argument("--path", help="Path of the directory to save data", default="./Data_MC")
parser.add_argument("-n", "--name", help="Name of the json file. Parameters are replaced by their values in default name.", default="Data_MC_noise_N_N_p_dt_theta.json")
parser.add_argument("--delete_file", help="Delete data file if the name are the same. Warning: turning this option to true will delete the data file without checking anything", action="store_true")
args = parser.parse_args()

def V(x):
    return x**4/4 - x**2/2


def dV(x):
    return x**3 - x


def ρ_st_white(x, m, β, θ):
    def f(y): return np.exp(-β*(V(y)+.5*θ*(y-m)**2))
    Z = integrate.quad(f, -np.inf, np.inf)
    return f(x)/Z[0]


def get_η_OU(N, N_p, dt):
    # -----Init-----
    Y = np.zeros((N_p, N))

    # -----Solving-----
    for n in range(N-1):
        Y[:, n+1] = Y[:, n] - Y[:, n]*dt + np.sqrt(2)*normal(0, np.sqrt(dt), N_p)

    return Y

def get_η_H(N, dt):
    Y = np.zeros((2, N+1))
    γ = 1
    A = np.array([[0, 1],
                  [-1, -γ]])
    D = np.array([[0, 0],
                  [0, np.sqrt(γ)]])
    y_η = np.array([1, 0])

    # -----Solving SDE-----
    for n in range(N):
        Y[:, n+1] = Y[:, n] + np.matmul(A, Y[:, n])*dt + \
            np.sqrt(2)*np.matmul(D, normal(0, np.sqrt(dt), 2))

    return np.matmul(y_η, Y)


def plotSimulation(X, N):
    # -----Plot-----
    # -----Fig-----
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    ax1.set_title("Histogram")
    ax2.set_title("Mean")
    writer1 = PillowWriter(fps=10)
    writer2 = PillowWriter(fps=10)

    with writer1.saving(fig1, f'histogram_β={β}_θ={θ}.gif', dpi=100):
        with writer2.saving(fig2, f'mean_β={β}_θ={θ}.gif', dpi=100):

            for n in np.arange(0, N, 10):
                ax1.clear()
                ax2.clear()
                _, bins, _ = ax1.hist(X[:, n+1], 100, density=True)
                ax1.plot(bins, ρ_st_white(bins, np.mean(X[:, n]), β, θ))
                ax2.plot(range(n+1), np.mean(X[:, :n+1], axis=0))
                writer1.grab_frame()
                writer2.grab_frame()


def SDEsolve(N, N_p, dt, θ, β, X_0, N_space):
    """
    Solving the SDE of interacting particles
    N_p: number of particles
    N: number of steps
    dt: size of time step
    sigma, beta, X_0: parameters
    """
    # -----Init-----
    X = [] # X is used for storing data and Y to solve SDE. Important for memory issue otherwise
    if args.noise == "OU":
        η = np.zeros(N_p)
        def μ_OU(Y): return -Y
        def σ_OU(Y): return np.sqrt(2)

    elif args.noise == "H":
        η = np.zeros(N_p)
        Z = np.zeros((2, N_p))
        γ = 1
        A = np.array([[0, 1],
                    [-1, -γ]])
        D = np.array([[0, 0],
                    [0, np.sqrt(γ)]])
        y_η = np.array([1, 0])

        def μ_H(Z): return np.matmul(A, Z)
        def σ_H(Z): return np.sqrt(2)*D

    sqrt_dt = np.sqrt(dt)
    Y = X_0
    def μ(Y):
        return -(dV(Y) + θ*(Y - np.mean(Y)))
    def σ(Y):
        return np.sqrt(2/β)
    
    # -----Solver-----
    for n in range(N):
        if n in N_space:
            # Saving every N_points
            X.append([int(n*dt), np.mean(Y)])

        if args.noise == "white":
            η = normal(0, 1/np.sqrt(dt), N_p) # Equivalent of normal(0, np.sqrt(dt), N_p) when multiplied by dt
        elif args.noise == "OU":
            η = η + μ_OU(η)*dt + σ_OU(η)*normal(0, sqrt_dt, N_p)
        elif args.noise == "H":
            Z += μ_H(Z)*dt + np.matmul(σ_H(Z), normal(0, np.sqrt(dt), (2, N_p)))
            η_old = η
            η = np.matmul(y_η, Z)
            η_new = η
            print(np.matmul(η_old, η_new))
            print(np.exp(-dt/2)*(np.cos(np.sqrt(3)/2 * dt) + np.sqrt(3)/3*np.sin(np.sqrt(3)/2 * dt)))
            print("############")

        Y = Y + µ(Y)*dt + σ(Y)*η*dt
    
    return X

def bifurcation_scheme(file_path, N, N_p, dt, θ, βs):
    """
    Creates the bifurcation scheme for coloured noise
    """
    # -----Init-----
    N_points = 10
    N_space = np.linspace(0, N-1, N_points, dtype=int)
    βs = checkFile(file_path, N, N_p, dt, θ, βs)
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # -----Solver-----
    print("#########")
    print("Start solving...")
    for β in βs:
        globals()["β"] = β
        X_0 = normal(0, np.sqrt(.1))
        M1 = SDEsolve(N, N_p, dt, θ, β, X_0, N_space)

        data["data_points"][f"{β}"] = M1

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

if __name__ == '__main__':

    # -----Init-----
    # PARAMETERS
    print("Initialisation of the parameters...")
    if args.name == "Data_MC_noise_N_N_p_dt_theta.json":
        FILE_NAME = f"Data_MC_{args.noise}_{args.N}_{args.N_p}_{args.dt}_{args.theta}.json"
    else:
        FILE_NAME = args.name
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    FILE_PATH = f"{args.path}/{FILE_NAME}"

    # Deleting file
    if args.delete_file and os.path.isfile(FILE_PATH):
        os.remove(FILE_PATH)

    N = args.N
    N_p = args.N_p
    dt = args.dt
    global θ
    θ = args.theta
    β_start = args.beta_start
    β_end = args.beta_end 
    N_β = args.N_beta
    step_β = args.step_beta

    if args.N_beta:
        βs = np.linspace(β_start, β_end, N_β)
    elif args.space_beta:
        βs = np.arange(β_start, β_end, step_β)

    bifurcation_scheme(FILE_PATH, N, N_p, dt, θ, βs)
