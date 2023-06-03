# %%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
By trying to solve the equation on the cumulants k_n  
"""

using DifferentialEquations
import PyPlot as plt
import JSON
using DataStructures
using LSODA

METHODS_IVP = [
    ["AutoTsit5_RK23", AutoTsit5(Rosenbrock23(autodiff=false))],
    ["Radau3", RadauIIA3(autodiff=false)],
    ["Radau5", RadauIIA5(autodiff=false)],
    ["FBDF", FBDF(autodiff=false)],
    ["LSODA", lsoda()]
    ]
ν = .5

function f!(dk, k, p, t)
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
    α, θ, σ_m, σ = p
    N = size(k)[1]
    _k = zeros(N+2)
    dk[:] .= 0
    _k[1:N] = k
    #B.C.: k_{N+1} = k_{N+2} = 0 

    dk[1] = θ*_k[1]
    dk[2] = σ^2  # it is not sigma² / 2 because we multiply by n=2

    # -----Definition of f-----
    """
    To define f, a sum from 1 to n-1 which contains the three sums will be computed
    Then the last terms of the sum are added
    And we suppose B.C.: k_{N+1} = k_{N+2} = 0
    """
    for n in 1:N
        Σ = sum([(
            σ_m^2/2*_k[i]*_k[n-i] / (factorial(big(i-1))*factorial(big(n-i-1)))
            - 3*_k[i]*_k[n-i+2] / (factorial(big(i-1))*factorial(big(n-i)))
            - sum([(
                _k[i]*_k[j]*_k[n+2-i-j] /
                (factorial(big(i-1))*factorial(big(j-1))*factorial(big(n-i-j+1)))
            ) for j in 1:n-i+1])
        ) for i in 1:n-1])

        dk[n] += n*(
            (
                α-θ+σ_m^2*(ν+(n-1)/2))*_k[n]
            - _k[n+2]
            + factorial(big(n-1))*(
                Σ
                - 3*_k[n]*_k[2]/factorial(big(n-1))
                - _k[n]*_k[1]^2/factorial(big(n-1))
            )
        )
    end
end

function SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, method)
    """
    Solving the cumulant ODE for 1 set of parameters
    """
    # -----Init-----
    p = (α, θ, σ_m, σ)
    k0 = zeros(N)
    k0[1] = 1
    k0[2] = 1.5^2
    
    # -----Solver-----
    prob = ODEProblem(f!, k0, (t0, t_end), p)
    alg = method
    k = solve(prob,alg)

    return k
end


function SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
    """
    Solving the cumulant ODE for 1 set of parameters
    And in stationnary state
    """
    # -----Init-----
    p = α, θ, σ_m, σ
    k0 = 0.01*ones(N)
    k0[0] = .5
    k0[1] = .5

    # -----Solver-----
    g(x) = f(0, x, α, θ, σ_m, σ) 
    k = root(g, k0)

    return k
end

function plotData(file_path)
    """
    File must be the same format as Data.json created
    """

    open(file_path, "r") do file
        global data = JSON.parse(file)
    end
    
    param = data["parameters"]
    methods, Ns, σs = param["methods"], param["Ns"], param["sigmas"] 

    for method in methods
        for N in Ns
            info = data[method]["$N"]
            M1s, time, success = info["points"], info["time"], info["success"]

            plt.scatter(σs[success.==1], M1s[success.==1], label="N=$N, time=$time s")

        end
        plt.xlabel("σ")
        plt.ylabel("m")
        plt.legend()
        plt.title("Mean with cumulant solving ODE with method '$method'")
        plt.savefig("Figs/$method.png")
        plt.close()
    end
end

if abspath(PROGRAM_FILE) == "/home/" || abspath(PROGRAM_FILE) == @__FILE__ 

    PLOT = false
    SOLVE = true
    FILE_NAME = "Data_k_julia.json"

    if SOLVE

        # -----Init-----
        Ns = [4, 8, 16, 24]
        α = 1
        θ = 4
        σ_m = .8
        N_σ = 2
        σs = LinRange(1.8, 2., N_σ)

        # Writing parameters
        data = OrderedDict(
            "parameters" => OrderedDict()
            )
        for (name, alg) in METHODS_IVP

            data[name] = OrderedDict(
                N => OrderedDict(
                    "points" => [],
                    "time" => 0,
                    "success" => [])
                for N in Ns)

        end
        data["parameters"] = OrderedDict(
            "methods" => [methods[1] for methods in METHODS_IVP],
            "Ns" => Ns,
            "sigmas" => σs
        )
        open(FILE_NAME, "w") do file
            local json_data = JSON.json(data, 4)
            write(file, json_data)
        end

        t0 = 0.0
        t_end = 10e4

        # -----Solving-----
        println("##########")
        println("Parameters: ")
        println("Ns=$Ns")
        println("First σ: $(σs[1]), Last σ: $(σs[end]), N_σ: $N_σ")
        println("##########")

        for (name, alg) in METHODS_IVP

            println("Solving with method: $name")
            for N in Ns

                # Init
                t1 = time()
                println("Solving for N=$N...")
                _data = data[name][N]

                for (i, σ) in enumerate(σs)

                    println(σ)
                    M1 = SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, alg)
                    success = SciMLBase.successful_retcode(M1)
                    append!(_data["success"],success)

                    if success
                        append!(_data["points"], M1[1, end])
                    else
                        append!(_data["points"], 0)
                        println("N=$N, σ=$σ, method=$name failed")
                    end

                    # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
                    # M1s.append(M1.x[0])

                end
                # Time
                t2 = time()
                _data["time"] = floor(Int, t2-t1)

                # Writing data
                open(FILE_NAME, "w") do file
                    local json_data = JSON.json(data, 4)
                    write(file, json_data)
                end
                

            end
        end
    end

    if PLOT
        file_path = "Data_julia.json"
        plotData(file_path)
    end

end