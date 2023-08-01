import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import distributionWhite as white
from findiff import FinDiff



def bifurcation_scheme(α, θ, Nσ, tol=1e-10):
    # -----Init-----
    fig, ax = plt.subplots()
    ms = []
    σ_crit = white.σ_c(θ, α, 0)
    σ_start = σ_crit - 0.1
    σ_end = σ_crit + 0.1
    σs = np.linspace(σ_start, σ_end, Nσ)

    for σ in σs:
        f = lambda m: white.R(θ, m, α, σ, 0) - m
        ε = +1 if σ < σ_crit else -1
        m_st = optimize.root_scalar(f, x0=1.0, x1=0.5).root# bracket=[ε*tol, 2.0])
        ms.append(m_st)

    ax.scatter(σs,ms)
    ax.plot([σ_crit, σ_crit], [np.max(ms), 0], color="RED")
    ax.text(σ_crit, np.max(ms), f"σ_c: {σ_crit:.3f}")
    ax.set_title("Self-consistency equation")
    ax.set_xlabel("σ")
    ax.set_ylabel("m")
    plt.show()
    return ms

def plotSelfConsistency(θ, α, σ_m):
    """
    Plots values of R(m) - m and R'(m) - 1
    Can be used to have boundaries around sigma_c and use it to have boundaries for root_scalar
    """
    # -----Init-----
    σ_crit = white.σ_c(θ, α, 0).root
    σ_start = σ_crit - 0.1
    σ_end = σ_crit + 0.1
    σs = np.linspace(σ_start, σ_end, 50)
    ms = np.linspace(-.5, .5, 50)
    fig, ax = plt.subplots()
    d_dm = FinDiff(0, ms[1]-ms[0])

    # -----Plot-----
    for _σ in σs:
        Rs = [white.R(θ, m, α, _σ, σ_m) - m for m in ms]
        dRs_dm = d_dm(np.array([white.R(θ, m, α, _σ, σ_m)-m for m in ms]))

        ax.set_xlabel("m")
        ax.plot(ms, 0*ms)
        ax.plot(ms, Rs, label="R(m)-m")
        ax.plot(ms, dRs_dm, label="R'(m)-1")
        ax.set_title("σ=%.3f" % _σ)
        ax.legend()
        plt.pause(0.01)
        ax.clear()

if __name__=='__main__':

    α = 1.0
    θ = 4.0 
    Nσ = 100

    bifurcation_scheme(α, θ, Nσ)
    # plotSelfConsistency(θ, α, 0)


