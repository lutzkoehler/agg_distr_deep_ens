## Simulation study: Script 1
# Generation of deep ensembles


import json
import logging
import os
import pickle
import time
from typing import Any, Tuple, Type

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from nptyping import Float, NDArray
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from sklearn.utils import resample

import BQNModels  # noqa: F401
import DRNModels  # noqa: F401
from BaseModel import BaseModel
from fn_basic import fn_upit

METHOD_CLASS_CONFIG = {
    "eva": "Eva",
    "mc_dropout": "Dropout",
    "dropconnect": "DropConnect",
    "variational_dropout": "VariationalDropout",
    "concrete_dropout": "ConcreteDropout",
    "bayesian": "Bayesian",
    "rand_init": "RandInit",
    "bagging": "RandInit",
    "batchensemble": "BatchEnsemble",
}
METHOD_NUM_MODELS = {
    "single_model": [
        "mc_dropout",
        "dropconnect",
        "variational_dropout",
        "bayesian",
        "concrete_dropout",
    ],
    "multi_model": [
        "rand_init",
        "bagging",
    ],
    "parallel_model": ["batchensemble"],
}


def run_ensemble_parallel_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    loss: list[Any],
    ens_method: str = "batchensemble",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Use one model to predict n_ens times in parallel

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
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

    # Set standard architecture
    if nn_deep_arch is None:
        nn_deep_arch = [[["Dense", 64], ["Dense", 32]]]

    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 100 * i_sim)

        ### Run model ###
        # Create model
        model = model_class(
            n_ens=n_ens,
            nn_deep_arch=nn_deep_arch[0],
            n_cores=num_cores,
            rpy_elements=rpy_elements,
            loss=loss,
        )

        # For BatchEnsemble:
        # If necessary, drop some observations to ensure correct batch size
        # Constraints:
        # 1. n_batch % n_ens = 0
        # 2. n_batch_rest % n_ens = 0 for n_train and n_valid
        if ens_method == "batchensemble":
            n_batch = model.hpar["n_batch"]
            # Resample missing datapoints: Train set
            n_train_to_add = n_ens - (X_train.shape[0] % n_batch) % n_ens
            train_indeces_to_add = np.random.choice(
                range(X_train.shape[0]), size=n_train_to_add, replace=True
            )
            X_train = np.vstack([X_train, X_train[train_indeces_to_add, :]])
            y_train = np.hstack([y_train, y_train[train_indeces_to_add]])
            # Resample missing datapoints: Validation set
            n_valid_to_add = n_ens - (X_valid.shape[0] % n_batch) % n_ens
            valid_indeces_to_drop = np.random.choice(
                range(X_valid.shape[0]), size=n_valid_to_add, replace=True
            )
            X_valid = np.vstack([X_valid, X_valid[valid_indeces_to_drop, :]])
            y_valid = np.hstack([y_valid, y_valid[valid_indeces_to_drop]])
            # Resample missing datapoints: Test set
            # Test resampling not necessary as implicit ensemble will
            # transformed to n_ens BatchEnsemble models each of n_ens = 1

            total_samples_added = n_train_to_add + n_valid_to_add
            log_message = (
                f"Added {total_samples_added} samples due to BatchEnsemble "
                f"(train: {n_train_to_add}, "
                f"valid: {n_valid_to_add})"
            )
            logging.warning(log_message)

        # Build model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        log_message = (
            f"{dataset.upper()}, {temp_nn.upper()}: Finished training of "
            f"{temp_nn}_sim_{i_sim}_ens_0.pkl "
            f"- {(model.runtime_est)/1e+9:.2f}s"
        )
        logging.info(log_message)

        # Take time
        start_time = time.time_ns()

        # Run all predictions
        model.predict(X_test=np.r_[X_valid, X_test])

        # Get results
        pred_nn_ls = model.get_results(y_test=np.hstack((y_valid, y_test)))

        # For-Loop over ensemble member
        for i_ens in range(n_ens):
            # Extract ensemble member prediction
            current_pred_nn = pred_nn_ls[i_ens]  # type: ignore

            # Transform ranks
            if "rank" in current_pred_nn["scores"].keys():
                current_pred_nn["scores"]["pit"] = fn_upit(
                    ranks=current_pred_nn["scores"]["rank"],
                    max_rank=max(current_pred_nn["scores"]["rank"]),
                )

                # Omit ranks
                current_pred_nn["scores"]["rank"] = np.nan

            # Take time
            end_time = time.time_ns()

            # Save ensemble member
            filename = os.path.join(
                f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl",  # noqa: E501
            )
            temp_data_out_path = os.path.join(data_out_path, filename)

            # Check for NaNs in predictions
            if np.any(np.isnan(current_pred_nn["f"])):
                log_message = f"NaNs predicted in {temp_data_out_path}"
                logging.error(log_message)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([current_pred_nn, y_valid, y_test], f)

            log_message = (
                f"{dataset.upper()}, {temp_nn.upper()}: "
                f"Finished prediction of {filename} - "
                f"{(end_time - start_time)/1e+9:.2f}s"
            )
            logging.info(log_message)

        del model


def run_ensemble_single_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    loss: list[Any],
    ens_method: str = "mc_dropout",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Use one model to predict n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
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

    # Set default list for multiple architectures
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
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 100 * i_sim)

        ### Run model ###
        # Create model
        model = model_class(
            n_ens=n_ens,
            nn_deep_arch=nn_deep_arch[0],
            n_cores=num_cores,
            rpy_elements=rpy_elements,
            training=training,  # Makes dropout active in testing
            loss=loss,
        )

        # Build model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        log_message = (
            f"{dataset.upper()}, {temp_nn.upper()}: Finished training of "
            f"{temp_nn}_sim_{i_sim}_ens_0.pkl "
            f"- {(model.runtime_est)/1e+9:.2f}s"
        )
        logging.info(log_message)

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
                    loss=loss,
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
                f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl",  # noqa: E501
            )
            temp_data_out_path = os.path.join(data_out_path, filename)

            # Check for NaNs in predictions
            if np.any(np.isnan(pred_nn["f"])):
                log_message = f"NaNs predicted in {temp_data_out_path}"
                logging.error(log_message)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([pred_nn, y_valid, y_test], f)

            log_message = (
                f"{dataset.upper()}, {temp_nn.upper()}: "
                f"Finished prediction of {filename} - "
                f"{(end_time - start_time)/1e+9:.2f}s"
            )
            logging.warning(log_message)

        del model


def run_ensemble_multi_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    loss: list[Any],
    ens_method: str = "rand_init",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Run and train a model type n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
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

    # Set standard architecture
    if nn_deep_arch is None:
        nn_deep_arch = [["Dense", 64], ["Dense", 32]]

    # Set default list for multiple architectures
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
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 100 * i_sim)

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
            # Select current model architecture
            temp_nn_deep_arch = nn_deep_arch[0]
            if i_ens in nn_deep_arch_ls:
                idx_arch = np.where(nn_deep_arch_ls == i_ens)[0][0]
                temp_nn_deep_arch = nn_deep_arch[idx_arch]

            # Create model
            model = model_class(
                nn_deep_arch=temp_nn_deep_arch,
                n_ens=n_ens,
                n_cores=num_cores,
                rpy_elements=rpy_elements,
                loss=loss,
            )

            # Build model
            model.fit(
                X_train=X_train_nn,
                y_train=y_train_nn,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            log_message = (
                f"{dataset.upper()}, {temp_nn.upper()}: "
                f"Finished training of {temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl - "
                f"{(model.runtime_est)/1e+9:.2f}s"
            )
            logging.info(log_message)

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
                f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl",  # noqa: E501
            )
            temp_data_out_path = os.path.join(data_out_path, filename)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([pred_nn, y_valid, y_test], f)

            log_message = (
                f"{dataset.upper()}, {temp_nn.upper()}: "
                f"Finished prediction of {filename} - "
                f"{(end_time - start_time)/1e+9:.2f}s"
            )
            logging.info(log_message)

            del model


def run_eva_multi_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    num_cores: int,
    loss: list[Any],
    ens_method: str = "mc_dropout",
    nn_deep_arch: list[Any] | None = None,
) -> None:
    """Reproduces MC dropout results from Walz et al. (2022)
    based on Gal & Ghahramani (2015).

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
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
    if i_sim < 17:
        return
    ### Initialization ###
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    # Read out class
    model_class = get_model_class("drn", "eva")

    ### Run grid search ###
    # !Grid search parameter are for boston housing dataset only!
    param_grid = {
        "p_dropout": [0.005, 0.01, 0.05, 0.1],
        "tau": [0.1, 0.15, 0.2],
    }

    best_p_dropout = 0
    best_tau = 0
    best_crps = float("inf")

    for p_dropout in param_grid["p_dropout"]:
        for tau in param_grid["tau"]:
            # Create model
            model = model_class(
                nn_deep_arch=[],
                n_ens=n_ens,
                n_cores=num_cores,
                rpy_elements=rpy_elements,
                p_dropout=p_dropout,
                tau=tau,
            )

            # Build model
            model.fit(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            # Make prediction
            model.predict(X_test=X_valid)

            # Get results
            pred_nn = model.get_results(y_test=y_valid)

            with open(
                f"{i_sim}_validation_results_gird_search.txt", "a"
            ) as myfile:
                myfile.write(
                    "Dropout_Rate: "
                    + repr(p_dropout)
                    + " Tau: "
                    + repr(tau)
                    + " :: "
                )
                myfile.write(
                    "RMSE: "
                    + repr(pred_nn["rmse"])
                    + " - CRPS: "
                    + repr(pred_nn["crps"])
                    + " - CRPS_E: "
                    + repr(pred_nn["crps_eva"])
                    + " - LogL: "
                    + repr(pred_nn["logl"])
                    + "\n"
                )

            if pred_nn["crps"] < best_crps:
                best_crps = pred_nn["crps"]
                best_tau = tau
                best_p_dropout = p_dropout

            log_message = f"Finished {i_sim} with {p_dropout} / {tau}"
            logging.info(log_message)

            del model
            del pred_nn

    # Fit best network
    model = model_class(
        nn_deep_arch=[],
        n_ens=n_ens,
        n_cores=num_cores,
        rpy_elements=rpy_elements,
        p_dropout=best_p_dropout,
        tau=best_tau,
    )

    # Build model
    model.fit(
        X_train=np.r_[X_train, X_valid],
        y_train=np.hstack((y_train, y_valid)),
        X_valid=X_valid,
        y_valid=y_valid,
    )

    # Make prediction
    model.predict(X_test=X_test)

    # Get results
    pred_nn = model.get_results(y_test=y_test)

    with open(f"{i_sim}_validation_results_gird_search.txt", "a") as myfile:
        myfile.write(
            "########################## "
            + "Best result Dropout_Rate: "
            + repr(best_p_dropout)
            + " Tau: "
            + repr(best_tau)
            + " :: "
        )
        myfile.write(
            "RMSE: "
            + repr(pred_nn["rmse"])
            + " - CRPS: "
            + repr(pred_nn["crps"])
            + " - CRPS_E: "
            + repr(pred_nn["crps_eva"])
            + " - LogL: "
            + repr(pred_nn["logl"])
            + "\n"
        )

    del model
    del pred_nn

    log_message = (
        f"Finished best model {i_sim} with {best_p_dropout} / {best_tau}"
    )
    logging.info(log_message)


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
    data_in_path: str, dataset: str, i_sim: int
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
    dataset : str
        Name of dataset
    i_sim : int
        Run number

    Returns
    -------
    tuple
        Contains (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    ### Simulated dataset ###
    if dataset.startswith("scen"):
        # Load corresponding data
        temp_data_in_path = os.path.join(data_in_path, f"sim_{i_sim}.pkl")

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
        if dataset.endswith("6"):
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

    ### UCI Dataset ###
    # Code used from https://github.com/yaringal/DropoutUncertaintyExps
    else:
        data = np.loadtxt(os.path.join(data_in_path, "data.txt"))
        index_features = np.loadtxt(
            os.path.join(data_in_path, "index_features.txt")
        )
        index_target = np.loadtxt(
            os.path.join(data_in_path, "index_target.txt")
        )

        # Separate features and target
        X = data[:, [int(i) for i in index_features.tolist()]]
        y = data[:, int(index_target.tolist())]

        # Get train and test split (i_sim)
        index_train = np.loadtxt(
            os.path.join(data_in_path, f"index_train_{i_sim}.txt")
        )
        index_test = np.loadtxt(
            os.path.join(data_in_path, f"index_test_{i_sim}.txt")
        )

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]

        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]

        # Add validation split
        num_training_examples = int(0.8 * X_train.shape[0])
        X_valid = X_train[num_training_examples:, :]
        y_valid = y_train[num_training_examples:]
        X_train = X_train[0:num_training_examples, :]
        y_train = y_train[0:num_training_examples]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    ### Initialize ###
    # Networks
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]
    nn_deep_arch = CONFIG["NN_DEEP_ARCH"]

    # Datasets considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulated runs
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Size of network ensembles
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Loss function "norm", "0tnorm", "tnorm"
    loss = CONFIG["PARAMS"]["LOSS"]

    # Get method for generating ensemble members
    ens_method = CONFIG["ENS_METHOD"]

    # Get number of cores for parallelization
    num_cores = CONFIG["NUM_CORES"]

    # Train a single model for each ensemble member
    if ens_method in METHOD_NUM_MODELS["single_model"]:
        run_ensemble = run_ensemble_single_model
    # Or use the same model to predict each ensemble member
    elif ens_method in METHOD_NUM_MODELS["multi_model"]:
        run_ensemble = run_ensemble_multi_model
    # Eva's MC dropout implementation
    elif ens_method == "eva":
        run_ensemble = run_eva_multi_model
    else:
        run_ensemble = run_ensemble_parallel_model

    # Generate grid with necessary information for each run
    run_grid = pd.DataFrame(
        columns=["dataset", "i_sim", "data_in_path", "data_out_path"]
    )
    for dataset in dataset_ls:
        data_in_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["INPUT_DIR"],
            dataset,
        )
        data_out_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["RESULTS_DIR"],
            dataset,
            CONFIG["ENS_METHOD"],
            CONFIG["PATHS"]["ENSEMBLE_F"],
        )
        if dataset.startswith("scen"):
            temp_n_sim = n_sim
        elif dataset in ["protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = 20
        for i_sim in range(temp_n_sim):
            new_row = {
                "dataset": dataset,
                "i_sim": i_sim,
                "data_in_path": data_in_path,
                "data_out_path": data_out_path,
            }
            run_grid = pd.concat(
                [run_grid, pd.DataFrame(new_row, index=[0])],
                ignore_index=True,
            )

    # Check for model and agg directories and create if necessary
    check_directories(run_grid=run_grid)

    # Run sequential or run parallel
    run_parallel = True

    if run_parallel:
        ### Run parallel ###
        Parallel(n_jobs=2, backend="multiprocessing")(
            delayed(run_ensemble)(
                dataset=row["dataset"],
                i_sim=row["i_sim"],
                n_ens=n_ens,
                nn_vec=nn_vec,
                data_in_path=row["data_in_path"],
                data_out_path=row["data_out_path"],
                num_cores=num_cores,
                ens_method=ens_method,
                nn_deep_arch=nn_deep_arch,
                loss=loss,
            )
            for _, row in run_grid.iterrows()
        )
    else:
        ### Run sequential ###
        for _, row in run_grid.iterrows():
            run_ensemble(
                dataset=row["dataset"],
                i_sim=row["i_sim"],
                n_ens=n_ens,
                nn_vec=nn_vec,
                data_in_path=row["data_in_path"],
                data_out_path=row["data_out_path"],
                num_cores=num_cores,
                ens_method=ens_method,
                nn_deep_arch=nn_deep_arch,
                loss=loss,
            )


def check_directories(run_grid) -> None:
    temp_path = run_grid["data_out_path"][0]
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)
        os.makedirs(temp_path.replace("model", "agg"))


if __name__ == "__main__":
    #### Deactivate GPU usage ####
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "5"

    #### Limit cores to use ####
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(3)

    ### Set log Level ###
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    main()
