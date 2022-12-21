## Simulation study: Script 1
# Generation of deep ensembles


import itertools
import json
import os
import pickle
import time
from typing import Any, Tuple, Type

import numpy as np
from joblib import Parallel, delayed
from nptyping import Float, NDArray
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from sklearn.utils import resample

import BQNModels  # noqa: F401
import DRNModels  # noqa: F401
from BaseModel import BaseModel, DropoutBaseModel
from fn_basic import fn_upit

METHOD_CLASS_CONFIG = {
    "standard_dropout": "Dropout",
    "mc_dropout": "Dropout",
    "rand_init": "RandInit",
    "bagging": "RandInit",
}
METHOD_NUM_MODELS = {
    "single_model": ["mc_dropout", "standard_dropout"],
    "multi_model": ["rand_init", "bagging"],
}


def run_ensemble_single_model(
    i_scenario: int,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    ens_method: str = "mc_dropout",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Use one model to predict n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    i_scenario : int
        Scenario / Model number
    i_sim : int
        Simulation run
    n_ens : int
        Ensemble size
    nn_vec : list[str]
        Contains NN types
    data_in_path : str
        Location of generated simulation data (see ss_0_data.py)
    data_out_path : str
        Location to save results
    ens_method : str
        Specifies the initialization method to use
    """
    ### Initialization ###
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }
    # Specify configs for different methods
    if ens_method == "mc_dropout":
        training = True
    else:
        training = False
    # Set standard architecture
    if nn_deep_arch is None:
        nn_deep_arch = [[["Dense", 64], ["Dense", 32]]]

    nn_deep_arch_ls = np.floor(
        np.linspace(start=0, stop=n_ens, num=len(nn_deep_arch) + 1)
    )
    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, i_scenario=i_scenario, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 10 * i_scenario + 100 * i_sim)

        ### Run model ###
        # Create model
        model = model_class(
            n_ens=n_ens,
            nn_deep_arch=nn_deep_arch[0],
            n_cores=num_cores,
            rpy_elements=rpy_elements,
            training=training,  # Makes dropout active in testing
        )

        # Build model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        # Update weights of standard dropout to p_dropout * weights
        if (ens_method == "standard_dropout") and isinstance(
            model, DropoutBaseModel
        ):
            model.scale_weights()

        # For-Loop over ensemble member
        for i_ens in range(n_ens):
            # Train new if model split
            if (i_ens != 0) and (i_ens in nn_deep_arch_ls):
                idx_arch = np.where(nn_deep_arch_ls == i_ens)[0][0]
                model = model_class(
                    n_ens=n_ens,
                    nn_deep_arch=nn_deep_arch[idx_arch],
                    n_cores=num_cores,
                    rpy_elements=rpy_elements,
                    training=training,  # Makes dropout active in testing
                )

                # Build model
                model.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                )

            # Take time
            start_time = time.time_ns()

            # Make prediction
            model.predict(X_test=np.r_[X_valid, X_test])

            # Get results
            pred_nn = model.get_results(y_test=np.hstack((y_valid, y_test)))

            if np.any(np.isnan(pred_nn["f"])):
                print("test")

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

            if np.any(np.isnan(pred_nn["f"])):
                print(temp_data_out_path)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([pred_nn, y_valid, y_test], f)

            print(
                f"{temp_nn.upper()}: Finished training of {filename}"
                f" - {(end_time - start_time)/1e+9}s",
            )

            if ens_method == "standard_dropout":
                i_ens += 1
                # Store predictions n_ens times
                while i_ens < n_ens:
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
                    i_ens += 1
                # Finish for loop
                break

        del model


def run_ensemble_multi_model(
    i_scenario: int,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    ens_method: str = "rand_init",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Run and train a model type n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    i_scenario : int
        Scenario / Model number
    i_sim : int
        Simulation run
    n_ens : int
        Ensemble size
    nn_vec : list[str]
        Contains NN types
    data_in_path : str
        Location of generated simulation data (see ss_0_data.py)
    data_out_path : str
        Location to save results
    ens_method : str
        Specifies the initialization method to use
    """
    ### Initialization ###
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }
    if nn_deep_arch is None:
        nn_deep_arch = [["Dense", 64], ["Dense", 32]]

    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, i_scenario=i_scenario, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 10 * i_scenario + 100 * i_sim)

        # For-Loop over ensemble member
        for i_ens in range(n_ens):
            # Take time
            start_time = time.time_ns()

            # Bagging
            # Draw sample with replacement of same size (5_000)
            X_train_nn: NDArray
            y_train_nn: NDArray
            if ens_method == "bagging":
                X_train_nn, y_train_nn = resample(
                    X_train,
                    y_train,
                    replace=True,
                    n_samples=X_train.shape[0],  # type: ignore
                )
            else:
                X_train_nn, y_train_nn = X_train, y_train

            ### Run model ###
            # Create model
            model = model_class(
                nn_deep_arch=nn_deep_arch,
                n_ens=n_ens,
                n_cores=num_cores,
                rpy_elements=rpy_elements,
            )

            # Build model
            model.fit(
                X_train=X_train_nn,
                y_train=y_train_nn,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            # Make prediction
            model.predict(X_test=np.r_[X_valid, X_test])

            # Get results
            pred_nn = model.get_results(y_test=np.hstack((y_valid, y_test)))

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

            del model


def get_model_class(temp_nn: str, ens_method: str) -> Type[BaseModel]:
    """Get model class based on string

    Parameters
    ----------
    temp_nn : str
        DRN / BQN / (HEN)

    Returns
    -------
    BaseModel
        Returns class that inherits from abstract class BaseModel
    """
    temp_nn_upper = temp_nn.upper()
    module = globals()[f"{temp_nn_upper}Models"]
    method = METHOD_CLASS_CONFIG[ens_method]
    model_class = getattr(module, f"{temp_nn_upper}{method}Model")

    return model_class


def train_valid_test_split(
    data_in_path: str, i_scenario: int, i_sim: int
) -> Tuple[
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
]:
    """Performs data split in train, validation and test set

    Parameters
    ----------
    data_in_path : str
        Location of simulated data
    i_scenario : int
        Scenario / Model number
    i_sim : int
        Run number

    Returns
    -------
    tuple
        Contains (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    # Load corresponding data
    temp_data_in_path = os.path.join(
        data_in_path, f"scen_{i_scenario}_sim_{i_sim}.pkl"
    )

    with open(temp_data_in_path, "rb") as f:
        (
            X_train,  # n_train = 6_000
            y_train,
            X_test,  # n_test = 10_000
            y_test,
            _,
            _,
        ) = pickle.load(f)

    # Indices of validation set
    if i_scenario == 6:
        i_valid = np.arange(start=2_500, stop=3_000, step=1)
    else:
        i_valid = np.arange(
            start=5_000, stop=6_000, step=1
        )  # n_valid = 1000 -> n_train = 5000

    # Split X_train/y_train in train and validation set
    X_valid = X_train[i_valid]  # length 1_000
    y_valid: NDArray[Any, Float] = y_train[i_valid]
    X_train = np.delete(arr=X_train, obj=i_valid, axis=0)  # length 5_000
    y_train = np.delete(arr=y_train, obj=i_valid, axis=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    # Path of simulated data
    data_in_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"], CONFIG["PATHS"]["SIM_DATA"]
    )

    # Path of deep ensemble forecasts
    data_out_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["ENS_METHOD"],
        CONFIG["PATHS"]["ENSEMBLE_F"],
    )

    ### Initialize ###
    # Networks
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]
    nn_deep_arch = CONFIG["NN_DEEP_ARCH"]

    # Models considered
    scenario_vec = CONFIG["PARAMS"]["SCENARIO_VEC"]

    # Number of simulated runs
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Size of network ensembles
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Get method for generating ensemble members
    ens_method = CONFIG["ENS_METHOD"]

    # Get number of cores for parallelization
    num_cores = CONFIG["NUM_CORES"]

    if ens_method in METHOD_NUM_MODELS["single_model"]:
        run_ensemble = run_ensemble_single_model
    else:
        run_ensemble = run_ensemble_multi_model

    ### Run sequential ###
    # for i_scenario in scenario_vec:
    #     for i_sim in range(n_sim):
    #         run_ensemble(
    #             i_scenario,
    #             i_sim,
    #             n_ens,
    #             nn_vec,
    #             data_in_path,
    #             data_out_path,
    #             num_cores,
    #             ens_method,
    #             nn_deep_arch=nn_deep_arch,
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
            num_cores=num_cores,
            ens_method=ens_method,
            nn_deep_arch=nn_deep_arch,
        )
        for i_scenario, i_sim in itertools.product(scenario_vec, range(n_sim))
    )


if __name__ == "__main__":
    main()
