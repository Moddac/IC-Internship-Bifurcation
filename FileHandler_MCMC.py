"""
File used to handle the data files created with Interacting_particles
The json file will be as followed:
{
"parameters": {
"N": N,
"N_p: N_p,
"dt": dt,
"theta": theta,
"betas": [beta_1, ..., beta_2],
},
"data_points": {
"beta_1": [(time, mean) ...],
"beta_2": [(time, mean) ...],
...
}
}
Number of {
"time"
"means"
} is decided by the user

Returns the betas that have not been computed in the data file
TODO: Having the last points if we want to continue simulation?
"""
import json
import numpy as np
import os


def createFile(file_path, N, N_p, dt, θ, γ, ε, βs):
    """
    Creates .json file if it doesn't exist, and initialize it
    """
    # -----Parameters-----
    data = {}
    data["parameters"] = {
        "N": N,
        "N_p": N_p,
        "dt": dt,
        "theta": θ,
        "gamma": γ,
        "epsilon": ε,
        "betas": list(βs)
    }
    data["data_points"] = {
        f"{β}": [
            # each tuple will be (time, mean)
        ]
        for β in βs}

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    return βs


def updateFile(file_path, N, N_p, dt, θ, γ, ε, βs):
    """
    Checking if the parameters are the same 
    and returnin the new betas that must be computed 
    """

    # -----Loading-----
    with open(file_path, "r") as file:
        data = json.load(file)
    param = data["parameters"]

    # -----Assert parameters-----
    # Checking if same parameters
    N, _N_p, _dt, _θ, _γ, _ε, _βs = param["N"], param["N_p"], param["dt"], param["theta"], param["gamma"], param["epsilon"], param["betas"]
    assert (N, _N_p, _dt, _θ, _γ, _ε) == (
        N, N_p, dt, θ, γ, ε), f"File with same name exist, but hyper parameters not the same. Please check values in file at {file_path}"
    
    βs_union = np.union1d(βs, _βs).tolist()
    new_βs = np.setdiff1d(βs, _βs).tolist()
    param["betas"] = βs_union

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    return new_βs


def checkFile(file_path, N, N_p, dt, θ, γ, ε, βs):
    """
    This checks if the data file already exists.
    It will search locally for a Data/file_name.json file and create
    folders and file if they don't exist

    If file already exists, checks which methods, N and sigmas have been done
    and update the file with new parameters

    Returns a matrix of index for every method and N to specify which sigma hasn't been computed

    N.B.: If the sigmas are updated, the computation will be for all the Ns (new and already done) for consistency
    """

    if not os.path.isfile(file_path):
        # If there is no file, create it and set parameters in it

        return createFile(file_path, N, N_p, dt, θ, γ, ε, βs)

    else:
        # If there is a file update it with new parameters

        return updateFile(file_path, N, N_p, dt, θ, γ, ε, βs)
