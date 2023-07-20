def all_partitions(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_partitions(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1,len(lst)):
            pair = (a,lst[i])
            for rest in all_partitions(lst[1:i]+lst[i+1:]):
                yield [pair] + rest

def Isserlis(n, E_XtXt, E_Xtηt):
    """
    IMPORTANT: do NOT use Isserlis over N=16, it takes way too long (because of all_partitions)
    """
    # Returns 0 if odd
    if (n+1)%2==1:
        return 0
    
    # -----Init-----
    L = n*["Xt"] + ["ηt"]
    Σ = 0

    for partition in all_partitions(L):
        Π = 1
        for pair in partition:
            if "ηt" in pair:
                Π *= E_Xtηt
            else:
                Π *= E_XtXt
        Σ += Π
    
    return Σ