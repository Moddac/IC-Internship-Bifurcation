import numpy as np
import pickle
from tools import Isserlis
from math import comb
from scipy.integrate import solve_ivp
from scipy.stats import norm
import time 

ζ = 1 / np.sqrt(2)

def f_moment(t, M, α, θ, σ_m, σ):
    """
    This is the function to solve dM/dt = f(M,t) 
    with M being a vector of function at a fix time t: M(t) = (M_1(t);...;M_N(t)) 
    """
    # -----Init-----
    N = M.shape[0]
    # _M represents the vector (M_0, M_1, ..., M_n, M_{n+1}, M_{n+2}, M_{-1}) with B.C on M_0, M_{n+1}, M_{n+2}, M_{-1}
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
    if t == 0:
        E_ηt = 0
        E_ηtηt = 0
    else:
        E_ηt = 0
        E_ηtηt = 1-np.exp(-2*t/(ε**2))
    M[0, 0] = 1
    M[1, 1] = u[N]  # X_t * η_t
    M[0, 1] = E_ηt
    M[1:N+1, 0] = u[:N]
    for n in range(2, N+1):
        M[n, 1] = np.sum([
            comb(n, k)*M[1, 0]**(n-k)*Isserlis(k,
                                               (M[2, 0]-M[1, 0]**2), (M[1, 1]-M[1, 0]*M[0, 1]))
            for k in range(n+1)])

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
            - M[i+2, 0]
            - M[i, 0]*(θ - α)
            + θ*M[1, 0]*M[i-1, 0]
            + ζ*σ*M[i-1, 1] / ε
        )
    F[N] = (
        - M[3, 1]
        - M[1, 1]*(θ - α)
        + θ*M[1, 0]*M[0, 1]
        + ζ*σ*E_ηtηt / ε
        - M[1, 1] / ε**2
    )
    
    return F

def SolveMoment_ODE(N, P, K, t0, t_end, α, θ, σ_m, σ, γ, ε, method, noise, STOP_TIME, mean=1, std=1):
    """
    Solving the cumulant ODE for 1 set of parameters
    Initial condition is a gaussian
    """
    # Loading bell functions
    global mn1, mn2
    fN1 = open('./functions/formula_mn1_'+str(N), "rb")
    fN2 = open('./functions/formula_mn2_'+str(N), "rb")
    mn1 = pickle.load(fN1)
    mn2 = pickle.load(fN2)
    fN1.close()
    fN2.close()

    if noise == "white":
        # -----Init-----
        M0 = [norm.moment(n, loc=mean, scale=std) for n in range(1, N+1)]

        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, α, θ, σ_m, σ):
            τ = (int(time.time()) - T)
            return int(τ < STOP_TIME)

        stop_time.terminal = True

        M = solve_ivp(f_moment, (t0, t_end), M0,
                      args=(α, θ, σ_m, σ), method=method, events=stop_time)

        return M.y[0, -1], (M.status==0)

    if noise == "OU":
        # -----Init-----
        u0 = [norm.moment(n, loc=mean, scale=std) for n in range(1, N+1)]
        u0 += [0]  # X_0 * η_0
        u0 = np.array(u0)

        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, α, θ, σ, ε):
            τ = (int(time.time()) - T)
            return int(τ < STOP_TIME)
        stop_time.terminal = True

        u = solve_ivp(f_moment_OU, (t0, t_end), u0,
                      args=(α, θ, σ, ε), method=method, events=stop_time)

        return u.y[0, -1], (u.status==0)
    
# def SolveMoment_Stationnary(N, α, θ, σ_m, σ):
#     """
#     Solving the cumulant ODE for 1 set of parameters
#     And in stationnary state
#     """
#     # -----Init-----
#     # N.B. HArd to find the right initial condition
#     M0 = .5*np.ones(N)
#     M0[5:7] = 1.5
#     M0[7:N] = [4*i for i in range(1, N-6)]

#     # -----Solver-----

#     def g(x): return f_moment(0, x, α, θ, σ_m, σ)
#     M = root(g, M0)

#     return M
#     # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
#     # M1s.append(M1.x[0])
    