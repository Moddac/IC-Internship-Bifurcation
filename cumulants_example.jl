"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems
"""

using SymPy
bell = sympy.functions.combinatorial.numbers.bell
using NLsolve
export nlsolve
import PyPlot as plt
using OffsetArrays
using BenchmarkTools
using ProfileView


function Σ(x,σ,σ_m)
    σ^2 + σ_m^2*x^2
end

function V_α(x,α)
    x^4/4 - α*x^2/2
end

function B(n,l,M)
    bell(n,l,M[1:n-l+2])
end

function f(M1,M2,N,α,θ,σ_m,σ)
    M = OffsetVector(zeros(N+4),-2)
    M[-1] = 0
    M[0] = 1
    M[1] = M1
    M[2] = M2

    for n in 1:N
        # α_r = α + σ_m*normal(0,.1)
        M[n+2] = (α - θ + n/2*σ_m^2)*M[n] + (n-1)/2*σ^2*M[n-2] + θ*M[1]*M[n-1]
    end

    return M
end

function cummulant(M1,M2,N,α,θ,σ_m,σ)

    M = f(M1,M2,N,α,θ,σ_m,σ)
    M_n1 = 0

    for l in range(2,N+2)
        M_n1 -= (-1)^(l-1)*factorial(l-1)*B(N+1,l,M)
    end

    M_n2 = (N+2)*M_n1*M[1]

    for k in range(2,N+1)
        M_n2 += .5*binomial(N+2,k)*M[k]*M[N+2-k]
    end
    for l in range(3,N+3)
        M_n2 -= (-1)^(l-1)*factorial(l-1)*B(N+2,l,M)
    end

    return Array{Float64}([M_n1,M_n2])
end

function bifuraction(N,α,θ,σ_m)

    σs = LinRange(.5,2.5,100)
    M1s = []
    for σ in σs 
        println("σ=$σ")
        g(M) = f(M[1],M[2],N,α,θ,σ_m,σ)[[N+1,N+2]] - cummulant(M[1],M[2],N,α,θ,σ_m,σ)
        root = nlsolve(g,[1/σ,1/σ]).zero[1]
        append!(M1s,root)
    end

    # plt.scatter(σs,M1s)
    # plt.show()
end

function main()
    N = 4
    α = 1
    θ = 4
    σ_m = .8
    @profview bifuraction(N,α,θ,σ_m)
end
    
if abspath(PROGRAM_FILE) == @__FILE__
    N = 4
    α = 1
    θ = 4
    σ_m = .8
    @profview bifuraction(N,α,θ,σ_m)
end