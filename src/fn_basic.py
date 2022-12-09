## Function file
# Helper functions

import numpy as np
from typing import Any


def fn_upit(ranks, max_rank):
    """Function to transform ranks to uPIT-values

    Parameters
    ----------
    ranks : n vector
        Ranks
    max_rank : positive integer
        Maximal rank

    Returns
    -------
    n vector
        uPIT values
    """
    ### Calculation ###
    # Transform to uPIT
    res = ranks / max_rank - np.random.uniform(
        low=0, high=1 / max_rank, size=len(ranks)
    )

    # Output
    return res


### Update hyperparameters ###
def update_hpar(
    hpar_ls: dict[str, Any], in_ls: dict[str, Any]
) -> dict[str, Any]:
    """Update hyperparameters

    Parameters
    ----------
    hpar_ls : Dict
        Default hyperparameter
    in_ls : Dict
        Selected hyperparameter given by user

    Returns
    -------
    Dict
        All hyperparameters including users selection
    """

    #### Initiation ####
    # Names of hyperparameters
    hpar_names = hpar_ls.keys()

    # Names of hyperparameter to update
    in_names = in_ls.keys()

    #### Update ####
    # Loop over given names
    for temp_hpar in in_names:
        # Update list if correct name is given
        if temp_hpar in hpar_names:
            hpar_ls[temp_hpar] = in_ls[temp_hpar]
        else:
            print(f"Wrong hyperparameter given: {temp_hpar}")

    #### Output ####
    # Return list
    return hpar_ls


# rm_const is only used in fn_nn_cs.py
# Thus, for now not necessary to implement.
"""
def rm_const(data: pd.DataFrame, cols=None, t_c=0):

    ### Initiation ###
    if cols is None:
        cols = data.columns

    ### Remove columns ###
    # Number of samples to check with
    n_check = min(10, data.shape[0])

    # Check on sample which rows are candidates (-> computational more
    # feasible)
    # bool_res <- (apply(data[sample(1:nrow(data), n_check),], 2, function(x)
    # sd(as.numeric(x)) ) <= t_c)
    random_indexes = np.random.randint(a=data.shape[0], size=n_check)
    bool_res = np.std(a=data[random_indexes, :], axis=0)
"""
