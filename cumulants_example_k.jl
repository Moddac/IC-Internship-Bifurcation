"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
Please see the paper to understand all the computations and parameteres
By trying to solve the equation on the cumulants k_n  
"""
# Imports
using DifferentialEquations
import PyPlot as plt
import JSON
using DataStructures
using LSODA

METHODS_IVP = [
    "AutoTsit5_RK23",
    "Radau5",
    "FBDF",
    "LSODA"
    ]

NAME_TO_ALG = Dict(
    "AutoTsit5_RK23" => AutoTsit5(Rosenbrock23(autodiff=false)),
    "Radau5" => RadauIIA5(autodiff=false),
    "FBDF" => FBDF(autodiff=false),
    "LSODA" => lsoda()
)

ν = .5

function f!(dk, k, p, t)
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
    (α, θ, σ_m, σ) = p
    N = size(k)[1]
    _k = zeros(N+2)
    #B.C.: k_{N+1} = k_{N+2} = 0 
    dk[:] .= 0
    _k[1:N] = k

    # First and second temrs are defined because of the kronecker symbol in the equations
    dk[1] = θ*_k[1]
    dk[2] = σ^2  

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

function SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, alg)
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
    k = solve(prob, alg)

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
    # Opening file to retrieve data
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
        plt.savefig("Figs/Cumulants_julia_$method.png")
        plt.close()
    end
end

function checkFile(file_name, methods, Ns, σs, α, θ, σ_m)
    """
    This checks if the data file already exists.
    It will search locally for a Data/file_name.json file and create
    folders and file if they don't exist

    If file already exists, checks which methods, N and sigmas have been done
    and update the file with new parameters

    Returns methods, Ns and a matrix of sigma to be computed, 
    according to those which have already been done and the new sigmas

    N.B.: If the sigmas are updated, the computation will be for all the Ns (new and already done) for consistency
    """

    # If directory does not exist, create it
    if !isdir("./Data")
        mkdir("./Data")
    end

    file_path = "./Data/$file_name"
    if !isfile(file_path)
        # If there is no file, create it and set parameters in it

        # Writing parameters
        data = Dict()
        data["parameters"] = Dict(
            "methods" => methods,
            "Ns" => Ns,
            "sigmas" => σs,
            "alpha" => α,
            "theta" => θ,
            "sigma_m" => σ_m,
        )

        for method in methods

            data["$method"] = Dict(
                "$N" => Dict(
                    "points" => [],
                    "time" => 0,
                    "success" => [])
                for N in Ns)

        end

        open(file_path, "w") do file
            write(file, JSON.json(data, 4))
        end

        # This matrix will contain the sigmas that have to be computed
        # after checking which ones have already been computed
        # Here the file was not created, so no sigma could have been computed
        σs_matrix = Dict()

        for method in methods
            σs_matrix["$method"] = Dict()
            σs_matrix_method = σs_matrix["$method"]

            for N in Ns
                σs_matrix_method["$N"] = σs
            end
        end

        return methods, Ns, σs_matrix

    else
        # If the file exists, check the Ns, sigmas and methods
        global data
        open(file_path, "r") do file
            global data = JSON.parse(file)
        end

        param = data["parameters"]
        _methods, _Ns, _σs = param["methods"], param["Ns"], param["sigmas"]

        σs_union = union(σs, _σs)
        # Values in sigmas but not in _sigmas
        new_σs = setdiff(σs, _σs)
        Ns_union = union(Ns, _Ns)
        methods_union = union(methods, _methods)

        # Updating file with new parameters
        param["Ns"] = Ns_union
        param["methods"] = methods_union
        param["sigmas"] = σs_union

        """
        For now: data points for a model are writen only if all the sigmas are done
        So we can check either if the list "points" is empty or not
        If empty: all the sigmas must be done
        If not empty: only the new sigmas
        TODO: write on the file sigma per sigma?
        """

        # This matrix will contain the sigmas that have to be computed
        # after checking which ones have already been computed
        σs_matrix = Dict()

        for method in methods_union
            σs_matrix["$method"] = Dict()
            σs_matrix_method = σs_matrix["$method"]

            if !(method in _methods)  # i.e. not defined in the file
                data["$method"] = Dict()
            end

            data_method = data["$method"]

            for N in Ns_union
                str_N = "$N"

                if !(str_N in keys(data_method))  # i.e. not defined in the file
                    data_method[str_N] = Dict(
                        "methods" => methods,
                        "Ns" => Ns,
                        "sigmas" => σs
                    )
                end

                if length(data_method[str_N]["points"]) == 0
                    # Case where N was not defined or was not done
                    σs_matrix_method[str_N] = σs_union

                else
                    # Else, only take the new sigmas
                    σs_matrix_method[str_N] = new_σs
                end
            end
        end

        open(file_path, "w") do file
            write(file, JSON.json(data, 4))
        end

        return methods_union, Ns_union, σs_matrix
    end

end

if (abspath(PROGRAM_FILE) == @__FILE__) || ("none" == @__FILE__) || true# none is for the Imperial cluster
    """
    This script is to solve the differential equations of the cumulants k_n
    for a dimension reduction of N.

    Please note: for the data writing of the points, the values are automatically
    sorted by decreasing order. It is not inconvenient as the function is supposed to be 
    deacreasing according to sigma, but one has to remember it. 
    """
    PLOT = false
    SOLVE = true
    FAIL_LIMIT = 3
    
    if SOLVE
        
        # -----Init-----
        Ns = [4, 8, 16, 24]
        α = 1
        θ = 4
        σ_m = .8
        N_σ = 4
        σs = LinRange(1.8, 2., N_σ)

        FILE_NAME = "Data_k_julia_alpha$(α)_theta$(θ)_sigma_m$(σ_m).json"

        methods, Ns, σs_matrix = checkFile(FILE_NAME,
                                           METHODS_IVP,
                                           Ns,
                                           σs,
                                           α,
                                           θ,
                                           σ_m)
        open("./Data/$FILE_NAME", "r") do file
            global data = JSON.parse(file)
        end

        t0 = 0.0
        t_end = 10e4

        # -----Solving-----
        println("##########")
        println("Parameters: ")
        println("Ns=$Ns")
        println("First σ: $(σs[1]), Last σ: $(σs[end]), N_σ: $N_σ")
        println("##########")

        for method in methods

            alg = NAME_TO_ALG[method]
            println("Solving with method: $method")
            for N in Ns

                # Init
                t1 = time()
                println("Solving for N=$N...")
                data_method = data[method]["$N"]
                σs = σs_matrix[method]["$N"]
                FAIL_COUNT = 0

                for (i, σ) in enumerate(σs)

                    if FAIL_COUNT < FAIL_LIMIT
                        M1 = SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ, alg)
                        success = SciMLBase.successful_retcode(M1)
                        append!(data_method["success"],success)

                        if success
                            append!(data_method["points"], M1[1, end])
                            FAIL_COUNT = 0
                        else
                            append!(data_method["points"], 0)
                            println("N=$N, σ=$σ, method=$method failed")
                            FAIL_COUNT += 1 
                        end

                    else
                        append!(data_method["success"], false)
                        append!(data_method["points"], 0)
                        if FAIL_COUNT == FAIL_LIMIT
                            println("Too much failure for N=$N, method=$method so skiping the value $N")
                            FAIL_COUNT += 1
                        end
                    end

                    # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
                    # M1s.append(M1.x[0])

                end
                # Time
                t2 = time()
                data_method["time"] += floor(Int, t2-t1)

                # Writing data
                open("./Data/$FILE_NAME", "w") do file
                    write(file, JSON.json(data, 4))
                end
                

            end
        end
    end

    if PLOT
        file_path = "Data_julia.json"
        plotData(file_path)
    end

end