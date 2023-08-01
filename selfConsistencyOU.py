"""
Tries to solve self consistency equation on m with a multiplicative function sigma
"""
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import distibutionOU as OU


def bifurcation_scheme(θ, τ, Nσ):
    # -----Init-----
    fig, ax = plt.subplots()
    ms = []
    σ_crit = OU.σ_c(τ, θ)
    σ_start = σ_crit - 0.1
    σ_end = σ_crit + 0.1
    σs = np.linspace(σ_start, σ_end, Nσ)

    for σ in σs:
        f = lambda m: OU.R(m, σ, τ, θ) - m
        # ε = +1 if σ < σ_crit else -1
        m_st = optimize.root_scalar(f, x0=1.0, x1=0.5)# bracket=[ε*tol, 2.0])
        ms.append(m_st.root)

    ax.scatter(σs,ms)
    ax.plot([σ_crit, σ_crit], [np.max(ms), 0], color="RED")
    ax.text(σ_crit, np.max(ms), f"σ_c: {σ_crit:.3f}")
    ax.set_title("Self-consistency equation")
    ax.set_xlabel("σ")
    ax.set_ylabel("m")
    plt.show()
    return ms

if __name__=='__main__':
    """
    Important: don't forget the factor ζ which makes data generated before a different  
    """ 
    ε = 0.001
    θ = 4
    τ = ε**2
    Nσ = 100

    bifurcation_scheme(θ, τ, Nσ)