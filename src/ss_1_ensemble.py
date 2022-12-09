## Simulation study: Script 1
# Generation of deep ensembles


import multiprocessing
import os
import pickle
import time
import itertools

import numpy as np
from typing import Any
from nptyping import NDArray, Float
from joblib import Parallel, delayed
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr


import fn_nn_ss
from fn_basic import fn_upit


def run_ensemble(
    i_scenario: int,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
) -> None:
    """Run different NN types for n_ens times and save results

    Parameters
    ----------
    i_scenario : int
        Scenario / Model number
    i_sim : int
        Simulation number
    n_ens : int
        Ensemble size
    nn_vec : list[str]
        Contains NN types
    data_in_path : str
        Location of generated simulation data (see ss_0_data.py)
    data_out_path : str
        Location to save results
    """
    ### Initialization ###
    # Choose number of cores
    num_cores = multiprocessing.cpu_count() - 1
    # if pc == "simu":
    #     num_cores = 10
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    ### Get data ###
    # Load corresponding data
    temp_data_in_path = os.path.join(
        data_in_path, f"scen_{i_scenario}_sim_{i_sim}.pkl"
    )

    X_train: NDArray[Any, Float]
    y_train: NDArray[Any, Float]
    X_test: NDArray[Any, Float]
    y_test: NDArray[Any, Float]
    with open(temp_data_in_path, "rb") as f:
        (
            X_train,  # n_train = 6000
            y_train,
            X_test,  # n_test = 10_000
            y_test,
            _,
            _,
        ) = pickle.load(f)

    # Indices of validation set
    if i_scenario == 6:
        i_valid = np.arange(start=2500, stop=3000, step=1)
    else:
        i_valid = np.arange(
            start=5000, stop=6000, step=1
        )  # n_valid = 1000 -> n_train = 5000

    # Observations of validation set
    y_valid: NDArray[Any, Float] = y_train[i_valid]

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out function
        fn_pp = getattr(fn_nn_ss, f"{temp_nn}_pp")

        # Set seed (same for each network variant)
        np.random.seed(123 + 10 * i_scenario + 100 * i_sim)

        # For-Loop over ensemble member
        for i_ens in range(n_ens):
            # Take time
            start_time = time.time_ns()

            # Postprocessing
            pred_nn: dict[str, Any] = fn_pp(
                train=X_train,
                test=np.r_[X_train[i_valid, :], X_test],
                y_train=y_train,
                y_test=np.hstack((y_valid, y_test)),
                i_valid=i_valid,
                n_ens=n_ens,
                n_cores=num_cores,
                rpy_elements=rpy_elements,
            )

            # Transform ranks
            if "rank" in pred_nn["scores"].keys():
                pred_nn["scores"]["pit"] = fn_upit(
                    ranks=pred_nn["scores"]["rank"],
                    max_rank=max(pred_nn["scores"]["rank"]),
                )

                # Omit ranks
                pred_nn["scores"]["rank"] = np.nan

            # Take time
            end_time = time.time_ns()

            # Save ensemble member
            filename = os.path.join(
                f"{temp_nn}_scen_{i_scenario}_sim_{i_sim}_ens_{i_ens}.pkl",  # noqa: E501
            )
            temp_data_out_path = os.path.join(data_out_path, filename)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([pred_nn, y_valid, y_test], f)

            print(
                f"{temp_nn.upper()}: Finished training of {filename}"
                f" - {(end_time - start_time)/1e+9}s",
            )


if __name__ == "__main__":
    # Path of simulated data
    data_in_path = os.path.join("data", "ss_data")

    # Path of deep ensemble forecasts
    data_out_path = os.path.join("data", "model")

    ### Initialize ###
    # Networks
    nn_vec = ["drn", "bqn"]  # For now without "hen"
    # nn_vec = ["bqn"]

    # Models considered
    # scenario_vec = range(1, 7, 1)
    scenario_vec = [1, 4]

    # Number of simulated runs
    # n_sim = 50
    n_sim = 10

    # Size of network ensembles
    # n_ens = 40
    n_ens = 10

    ### Run sequential ###
    # for i_scenario in scenario_vec:
    #     for i_sim in range(n_sim):
    #         run_ensemble(
    #             i_scenario, i_sim, n_ens, nn_vec, data_in_path, data_out_path
    #         )

    ### Run parallel ###
    Parallel(n_jobs=7, backend="multiprocessing")(
        delayed(run_ensemble)(
            i_scenario=i_scenario,
            i_sim=i_sim,
            n_ens=n_ens,
            nn_vec=nn_vec,
            data_in_path=data_in_path,
            data_out_path=data_out_path,
        )
        for i_scenario, i_sim in itertools.product(scenario_vec, range(n_sim))
    )
