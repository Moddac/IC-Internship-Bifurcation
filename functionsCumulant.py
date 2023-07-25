import numpy as np
from math import factorial
from scipy.integrate import solve_ivp
import time

ν = .5
ζ = 1 / np.sqrt(2)

def f_cumulant(t, k, N, α, θ, σ_m, σ):
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
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
            (α-θ+σ_m**2*(ν+(n-1)/2))*_k[n]
            - _k[n+2]
            + factorial(n-1)*(
                Σ
                - 3*_k[n]*_k[2]/factorial(n-1)
                - _k[n]*_k[1]**2/factorial(n-1)
            )
        )

    return F


def f_cumulant_OU(t, κ, N, P, α, θ, σ, ε):
    """
    Function for the ODE system of kappa for a OU noise
    """
    # -----Init-----
    _κ = np.zeros((N+3, P+2))
    _κ[:N+1, :P+1] = np.reshape(κ, (N+1, P+1), order='F')

    # -----Definition of f-----
    F = np.zeros((N+1, P+1))

    for n in range(N+1):
        for p in range(P+1):
            Σ = -factorial(n)*factorial(p)*np.sum([

                3*_κ[α_x, α_η]*_κ[n-α_x+2, p-α_η] /
                (factorial(α_x-1)*factorial(n-α_x)*factorial(α_η)*factorial(p-α_η))  \
                + np.sum([

                    _κ[α_x, α_η]*_κ[β_x, β_η]*_κ[n-α_x-β_x+2, p-α_η-β_η] /
                    (factorial(α_x-1)*factorial(β_x-1)*factorial(n-α_x-β_x+1)*factorial(α_η)*factorial(β_η)*factorial(p-α_η-β_η))

                for β_x in range(1,n-α_x+2) for β_η in range(p-α_η+1)])

            for α_x in range(1, n+1) for α_η in range(p+1)])

            F[n, p] = (
                -n*_κ[n+2, p]
                +Σ
                +n*(α-θ)*_κ[n,p]
                +θ*_κ[1,0]*(n==1 and p==0)
                +ζ*n*σ*_κ[n-1, p+1] / ε
                -p*_κ[n,p] / ε**2
                +2*(n==0 and p==2) / ε**2
            )
    return F.flatten('F')

def f_cumulant_OU_inertia(t, κ, N, M, K, α, θ, σ, γ, ε):
    """
    Function for the ODE system of kappa for a OU noise with inertia
    """
    # -----Init-----
    _κ = np.zeros((N+4, M+2, K+2))
    _κ[:N+1, :M+1, :K+1] = np.reshape(κ, (N+1, M+1, K+1), order='F')

    # -----Definition of f-----
    F = np.zeros((N+1, M+1, K+1))
    for n in range(N+1):
        for m in range(M+1):
            for k in range(K+1):
                Σ = factorial(n)*factorial(m)*factorial(k)*np.sum([

                    3*_κ[α_q+2, α_p, α_η]*_κ[n-α_q+1, m-α_p-1, k-α_η] /
                    (factorial(α_q)*factorial(α_p)*factorial(α_η)*factorial(n-α_q)*factorial(m-α_p-1)*factorial(k-α_η))\
                    + np.sum([

                        _κ[α_q+1, α_p, α_η]*_κ[β_q+1, β_p, β_η]*_κ[n-α_q-β_q+1, m-α_p-β_p-1, k-α_η-β_η] /
                        (factorial(α_q)*factorial(α_p)*factorial(α_η)*factorial(β_q)*factorial(β_p)*factorial(β_η)*factorial(n-α_q-β_q)*factorial(m-α_p-β_p-1)*factorial(k-α_η-β_η))

                    for β_q in range(n-α_q+1) for β_p in range(m-α_p) for β_η in range(k-α_η+1)])

                for α_q in range(n+1) for α_p in range(m) for α_η in range(k+1)])

                F[n, m, k] = γ*(
                    +n*_κ[n-1, m+1, k]
                    -m*_κ[n+3, m-1, k]
                    -Σ
                    +m*(1.-θ)*_κ[n+1, m-1, k]
                    +θ*_κ[1, 0, 0]*((n,m,k)==(0,1,0))
                    -γ*m*_κ[n, m, k]
                    +ζ*m*σ*_κ[n, m-1, k+1] / ε
                ) + (-k*_κ[n, m, k]+2*((n,m,k)==(0,0,2))) / ε**2

    # print("Mean:",_κ[1,0,0])
    # print("Eta:",_κ[0,0,2])
    return F.flatten('F')


def SolveCumulant_ODE(N, P, K, t0, t_end, α, θ, σ_m, σ, γ, ε, method, noise, STOP_TIME):
    """
    Solving the cumulant ODE for 1 set of parameters
    """

    if noise == "white": # White noise
        # -----Init-----
        κ0 = np.zeros(N)
        κ0[0] = 1  # mean
        κ0[1] = 1.5**2  # variance

        # -----Solver-----
        κ = solve_ivp(f_cumulant, (t0, t_end), κ0,
                      args=(N, α, θ, σ_m, σ), method=method)

        return κ.y[0, -1], (κ.status==0)

    if noise == "OU" and γ==0.: # OU noise without inertia
        # -----Init-----
        κ0 = np.zeros((N+1, P+1))
        κ0[1, 0] = 1
        κ0[2, 0] = 1.5**2
        κ0 = κ0.flatten('F')

        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, N, P, α, θ, σ, ε):
            τ = (int(time.time()) - T)
            return int(τ < STOP_TIME)
        stop_time.terminal = True

        κ = solve_ivp(f_cumulant_OU, (t0, t_end), κ0,
                      args=(N, P, α, θ, σ, ε), method=method, events=stop_time)
        
        return κ.y[1, -1], (κ.status==0)
    
    if noise == "OU" and γ!=0.: # OU noise with inertia
        # -----Init-----
        κ0 = np.zeros((N+1, P+1, K+1))
        κ0[1, 0, 0] = 1
        κ0[2, 0, 0] = 1.5**2
        κ0 = κ0.flatten('F')

        # -----Solver-----
        T = int(time.time()) - 1

        def stop_time(t, y, N, P, K, α, θ, σ, γ, ε):
            τ = (int(time.time()) - T)
            return int(τ < STOP_TIME)
        # stop_time.terminal = True

        κ = solve_ivp(f_cumulant_OU_inertia, (t0, t_end), κ0,
                      args=(N, P, K, α, θ, σ, γ, ε), method=method, events=stop_time)
        return κ.y[1, -1], (κ.status==0)

    
# def SolveCumulant_Stationnary(N, α, θ, σ_m, σ):
#     """
#     Solving the cumulant ODE for 1 set of parameters
#     And in stationnary state
#     """
#     # -----Init-----
#     k0 = 0.01*np.ones(N)
#     k0[0] = .5
#     k0[1] = .5

#     # -----Solver-----
#     def g(x): return f_cumulant(0, x, α, θ, σ_m, σ)
#     k = root(g, k0)

#     return k