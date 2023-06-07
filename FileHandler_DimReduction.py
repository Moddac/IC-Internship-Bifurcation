"""
File used to handle the data files created with Dim_reduction.py
"""
import json
import numpy as np
import os

def createFile(file_path, methods, Ns, σs, α, θ, σ_m):
    """
    Creates .json file if it doesn't exist, and initialize it
    """
    # -----Parameters-----
    data = {}
    data["parameters"] = {
        "methods": methods,
        "Ns": Ns,
        "sigmas": list(σs),
        "alpha": α,
        "theta": θ,
        "sigma_m": σ_m,
    }

    for method in methods:
        data[f"{method}"] = {
            f"{N}": {
                "points": [],
                "time": 0,
                "success": []
            }
            for N in Ns}

    # -----Writing parameters-----
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    # This matrix will contain the index of sigmas that have to be computed
    # after checking which ones have already been computed
    # Here the file was not created, so no sigma could have been computed
    indx_matrix = {}
    full_indx = list(range(len(σs)))

    for method in methods:
        indx_matrix[f"{method}"] = {}
        indx_matrix_method = indx_matrix[f"{method}"]

        for N in Ns:
            indx_matrix_method[f"{N}"] = full_indx

    return indx_matrix


def updateFile(file_path, methods, Ns, σs, α, θ, σ_m):
    """
    Update existing .json file with the new parameters
    """
    # -----Loading-----
    # If the file exists, check the Ns, sigmas and methods
    with open(file_path, "r") as file:
        data = json.load(file)
    param = data["parameters"]

    # -----Assert parameters-----
    # Checking if same hyper parameters
    _α, _θ, _σ_m = param["alpha"], param["theta"], param["sigma_m"]
    assert (_α, _θ, _σ_m) == (
        α, θ, σ_m), f"File with same name exist, but hyper parameters not the same. Please check values in file at {file_path}"

    _methods, _Ns, _σs = param["methods"], param["Ns"], param["sigmas"]

    # -----New parameters-----
    σs_union = np.union1d(σs, _σs).tolist()
    print(σs_union)
    # Values in sigmas but not in _sigmas
    new_σs = np.setdiff1d(σs, _σs).tolist()
    Ns_union = np.union1d(Ns, _Ns).astype(int).tolist()
    methods_union = np.union1d(methods, _methods).tolist()

    # Updating file with new parameters
    param["Ns"] = Ns_union
    param["methods"] = methods_union
    param["sigmas"] = σs_union

    """
    For now: data points for a model are writen only if all the sigmas are done
    So we can check either if the list "points" is empty or not
    If empty: all the sigmas must be done
    If not empty: only the new sigmas

    new methods and N are also initialized in the data here.
    TODO: write on the file sigma per sigma?
    """

    # This matrix will contain the index of sigmas that have to be computed
    # after checking which ones have already been computed
    indx_matrix = {}
    full_indx = list(range(len(σs)))
    indx_new_σs = np.nonzero([σ in new_σs for σ in σs_union])[0]

    for method in methods_union:
        indx_matrix[f"{method}"] = {}
        indx_matrix_method = indx_matrix[f"{method}"]

        if method not in _methods:  # i.e. not defined in the file, so create it
            data[f"{method}"] = {}
        data_method = data[f"{method}"]

        for N in Ns_union:
            str_N = f"{N}"
            if str_N not in data_method.keys():  # i.e. not defined in the file
                data_method[str_N] = {
                    "points": [],
                    "time": 0,
                    "success": []
                }

            if len(data_method[str_N]["points"]) == 0:
                # Case where N was not defined or was not done
                indx_matrix_method[str_N] = full_indx

            else:
                # Else, only take the new sigmas
                indx_matrix_method[str_N] = indx_new_σs

    # -----Writing new paramters-----
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    return indx_matrix


def checkFile(file_path, methods, Ns, σs, α, θ, σ_m):
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

        return createFile(file_path, methods, Ns, σs, α, θ, σ_m)

    else:
        # If there is a file update it with new parameters

        return updateFile(file_path, methods, Ns, σs, α, θ, σ_m)