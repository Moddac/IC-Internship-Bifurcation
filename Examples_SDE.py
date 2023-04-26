"""
Showing how to use 'Stochastic_Integral.py' by solving multiple equations with known solutions
"""
if __name__=='__main__':

    import numpy as np
    from numpy.random import normal
    from Stochastic_Integral import SDEModel

    """
    Brownian motion
    dX_t = sqrt(2σ)dW_t, X_0 = x
    Solution = X_t = x + sqrt(2σ)W_t
    """
    #-----PARAMETERS-----
    T = 1
    N = 1_000
    σ = 0.1
    X_0 = 0

    def µ_Brownian(X,B,t,i):
        return 0
    def σ_Brownian(X,B,t,i):
        return np.sqrt(2*σ) 
    SDE_Brownian = SDEModel(µ_Brownian,σ_Brownian)

    def p_ref_Brownian(t):
        return X_0 + np.sqrt(2*σ)*normal(0,np.sqrt(t))
    def traj_ref_Brownian(t):
        return X_0

    # reference_solution_Brownian = {"proba": p_ref_Brownian, "traj": traj_ref_Brownian}
    # SDE_Brownian.solve(T=1,N_p=1,X_0=X_0,reference_solution=reference_solution_Brownian,PDF=True,trajectory=True,nb_simul=100_000)

    #####################################################################################################################################################################################

    """
    Ornstein_Uhlenbeck 
    dX_t = -αX_tdt + sqrt(2σ)dW_t
    X_0 = x
    Solution: X_t = exp(-αt)x + sqrt(2σ)int_0^t exp(-α(t-s))dW_s
    """  
    #-----PARAMETERS-----
    T = 1
    N = 1_000
    σ = 0.1
    α = 0.2
    X_0 = 1

    def µ_OU(X,B,t,i):
        return -α*X
    def σ_OU(X,B,t,i):
        return np.sqrt(2*σ)
    SDE_OU = SDEModel(µ_OU,σ_OU)

    """
    In the solution there is a stochastic integral
    To model it, we use an SDE wih µ=0 and deterministic variance
    """
    def µ_int(X,B,t,i):
        return 0
    def σ_int(X,B,t,i):
        return np.exp(α*t)
    SDE_int = SDEModel(µ_int,σ_int)

    def p_ref_OU(t):
        return np.exp(-α*t)*X_0 + np.sqrt(2*σ)*np.exp(-α*t)*SDE_int.simul(t,N=1000,X_0=0,N_p=1)[0,-1]
    def traj_ref_OU(t):
        return np.exp(-α*t)*X_0
    
    reference_solution_OU = {"proba": p_ref_OU, "traj": traj_ref_OU}
    SDE_OU.solve(T=1,N_p=1,X_0=X_0,reference_solution=reference_solution_OU,PDF=True,trajectory=True,nb_simul=100_000)

    #####################################################################################################################################################################################

    """
    Geometric Brownian Motion
    dX_t = µX_tdt + σX_tdW_t
    X_0 = x
    Solution: X_t = xexp((µ-σ²/2)t + σW_t) 
    """
    
    #-----Parameters-----
    T = 1
    N = 1_000
    σ = .1
    µ = .5
    X_0 = 1

    def µ_GBM(X,B,t,i):
        return µ*X
    def σ_GBM(X,B,t,i):
        return σ*X
    SDE_GBM = SDEModel(µ_GBM,σ_GBM)

    def p_ref_GBM(t):
        return X_0*np.exp((µ-σ**2/2)*t + σ*normal(0,np.sqrt(t)))
    def traj_ref_GBM(t):
        return X_0*np.exp((µ-σ**2/2)*t)
    
    reference_solution_GBM = {"proba": p_ref_GBM, "traj": traj_ref_GBM}
    SDE_GBM.solve(T=1,N_p=1,X_0=X_0,reference_solution=reference_solution_GBM,PDF=True,trajectory=True,nb_simul=100_000)