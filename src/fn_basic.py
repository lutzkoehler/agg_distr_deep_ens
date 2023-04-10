## Function file
# Helper functions

import logging
from typing import Any

import numpy as np

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


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
            log_message = f"Wrong hyperparameter given: {temp_hpar}"
            logging.error(log_message)

    #### Output ####
    # Return list
    return hpar_ls
