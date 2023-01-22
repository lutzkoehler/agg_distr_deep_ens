## Simulation study: Script 2
# Aggregation of deep ensembles

# import concurrent.futures
import json
import os
import pickle
from functools import partial

# from multiprocessing.pool import Pool
from time import time_ns

import numpy as np
import pandas as pd
import scipy.stats as ss
from joblib import Parallel, delayed
from rpy2.robjects import default_converter, numpy2ri, vectors
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.optimize import minimize

from fn_basic import fn_upit
from fn_eval import bern_quants, fn_scores_distr, fn_scores_ens


### Functions for weight estimation (and uPIT) ###
def fn_vi_drn(a: float, w: float, f_sum, y: np.ndarray, **kwargs) -> float:
    """CRPS of VI forecast in case of DRN

    Parameters
    ----------
    a : scalar
        Intercept
    w : positive scalar
        Equal weight
    f_sum : n_train x 2 matrix
        Sum of single DRN forecasts
    y : n_train vector
        Observations

    Returns
    -------
    float
        CRPS of VI forecast
    """
    ### Initiation ###
    # Penalty for non-positive weights
    if w <= 0:
        return 1e6

    ### Calculation ###
    # Calculate weighted average
    f = w * f_sum

    # Add intercept term (only to location)
    f[:, 0] = a + f[:, 0]

    # Calculate CRPS
    scoring_rules = importr("scoringRules")
    np_cv_rules = default_converter + numpy2ri.converter
    y_vector = vectors.FloatVector(y)
    with localconverter(np_cv_rules) as cv:  # noqa: F841
        res = np.mean(
            scoring_rules.crps_norm(y=y_vector, mean=f[:, 0], sd=f[:, 1])
        )

    return res


def fn_vi_bqn(a, w, f_sum, y, **kwargs):
    """CRPS of VI forecast in case of BQN

    Parameters
    ----------
    a : scalar
        Intercept
    w : positive scalar
        Weight
    f_sum : n_train x 2 matrix
        Sum of single BQN coefficients
    y : n_train vector
        Observations
    q_levels : vector
        Quantile levels for evaluation

    Returns
    -------
    scalar
        CRPS of VI forecast
    """
    ### Initiation ###
    # penalty for non-positive weights
    if w <= 0:
        return 1e6

    ### Calculation ###
    # Calculate weighted average
    alpha = w * f_sum

    # Calculate quantiles (adds a to every entry)
    q = a + bern_quants(alpha=alpha, q_levels=kwargs["q_levels"])

    # Calculate CRPS
    scoring_rules = importr("scoringRules")
    np_cv_rules = default_converter + numpy2ri.converter
    y_vector = vectors.FloatVector(y)
    with localconverter(np_cv_rules) as cv:  # noqa: F841
        res = np.mean(scoring_rules.crps_sample(y=y_vector, dat=q))

    return res


def fn_apply_bqn(i, n_ens, **kwargs):
    # Sample individual distribution
    i_rep = np.random.choice(
        a=range(n_ens), size=kwargs["n_lp_samples"], replace=True
    )  # with replacement

    # Draw from individual distributions
    alpha_vec = [
        bern_quants(
            alpha=kwargs["f_ls"][f"alpha{j}"][i, :],
            q_levels=np.random.uniform(size=1),
        )
        for j in i_rep
    ]

    return np.reshape(np.asarray(alpha_vec), newshape=(kwargs["n_lp_samples"]))


### Parallel-Function ###
def fn_mc(
    temp_nn: str, dataset: str, n_ens: int, i_sim: int, **kwargs
) -> None:
    """Function for parallel computing

    Parameters
    ----------
    temp_nn : str
        Network type
    dataset : str
        Name of dataset
    n_ens : integer
        Size of network ensemble
    i_sim : integer
        Number of simulation

    Returns
    -------
    None
        Aggregation results are saved to data/agg
    """
    ### Initialization ###
    # Create pool for parallel processing
    # current_pool = Pool(processes=kwargs["num_cores"])

    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    # Get aggregation methods
    agg_meths = kwargs["agg_meths_ls"][temp_nn]

    # Create list for ensemble member forecasts on test and validation set
    f_ls = {}
    f_valid_ls = {}

    ### Get data ###
    # For-Loop over ensemble member
    y_test = []
    for i_ens in range(n_ens):
        # Load ensemble member
        filename = f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl"
        temp_data_in_path = os.path.join(kwargs["data_in_path"], filename)
        with open(temp_data_in_path, "rb") as f:
            pred_nn, y_valid, y_test = pickle.load(f)

        # Get indices of validation and test set
        i_valid = list(range(len(y_valid)))
        i_test = [x + len(y_valid) for x in range(len(y_test))]

        # Save forecasts
        if temp_nn == "drn":
            # Validation set
            f_valid_ls[f"f{i_ens}"] = pred_nn["f"][i_valid, :]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
        elif temp_nn == "bqn":
            # Validation set
            f_valid_ls[f"f{i_ens}"] = pred_nn["f"][i_valid, :]
            f_valid_ls[f"alpha{i_ens}"] = pred_nn["alpha"][i_valid, :]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
            f_ls[f"alpha{i_ens}"] = pred_nn["alpha"][i_test, :]
        elif temp_nn == "hen":
            pass

    # Average forecasts depending on method
    f_sum = np.empty(shape=f_ls["f0"].shape)
    f_valid_sum = np.empty(shape=f_valid_ls["f0"].shape)
    alpha_sum = np.empty(shape=f_ls["f0"].shape)
    if temp_nn == "drn":
        # Average parameters
        # Sum mean and sd for each obs over ensembles
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # Calculate sum of validation set forecasts
        f_valid_sum = np.asarray(
            sum([f_valid_ls[key] for key in f_valid_ls.keys() if "f" in key])
        )
    elif temp_nn == "bqn":
        # Average forecasts
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # Average coefficients
        alpha_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "alpha" in key])
        )

        # Calculate sum of validation set forecasts
        f_valid_sum = np.asarray(
            sum(
                [
                    f_valid_ls[key]
                    for key in f_valid_ls.keys()
                    if "alpha" in key
                ]
            )
        )
    elif temp_nn == "hen":
        pass

    ### Aggregation ###
    # For-Loop over aggregation methods
    for temp_agg in agg_meths:
        # Create list
        pred_agg = {}

        # Take time
        start_time = time_ns()

        # Different cases
        if (temp_nn == "drn") & (temp_agg == "lp"):
            # Function for mixture ensemble
            def fn_apply_drn(i, **kwargs):
                # Sample individual distribution
                i_rep = np.random.choice(
                    a=range(n_ens), size=kwargs["n_lp_samples"], replace=True
                )  # with replacement

                # Get distributional parameters
                temp_f = np.asarray(
                    [kwargs["f_ls"][f"f{j}"][i, :] for j in i_rep]
                )

                # Draw from individual distribution
                res = ss.norm.rvs(
                    size=kwargs["n_lp_samples"],
                    loc=temp_f[:, 0],
                    scale=temp_f[:, 1],
                )

                # Output
                return np.asarray(res)

            # Simulate ensemble for mixture
            pred_agg["f"] = np.asarray(
                [
                    fn_apply_drn(i=i, **dict(kwargs, f_ls=f_ls))
                    for i in range(len(y_test))
                ]
            )

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"], y=y_test, rpy_elements=rpy_elements
            )

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_lp_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan
        elif (temp_nn == "drn") & (temp_agg == "vi"):
            # Average parameters
            pred_agg["f"] = f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "drn") & (temp_agg == "vi-w"):
            # Wrapper function
            def fn_optim_drn_vi_w(x):
                return fn_vi_drn(a=0, w=x, f_sum=f_valid_sum, y=y_valid)

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "drn") & (temp_agg == "vi-a"):
            # Wrapper function
            def fn_optim_drn_vi_a(x):
                return fn_vi_drn(
                    a=x, w=1 / n_ens, f_sum=f_valid_sum, y=y_valid
                )

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_a,
                x0=0,
                method="Nelder-Mead",
            )

            # Read out intercept
            pred_agg["a"] = est.x

            # Calculate equally weighted VI
            pred_agg["f"] = f_sum / n_ens

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "drn") & (temp_agg == "vi-aw"):
            # Wrapper function
            def fn_optim_drn_vi_aw(x):
                return fn_vi_drn(a=x[0], w=x[1], f_sum=f_valid_sum, y=y_valid)

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "bqn") & (temp_agg == "lp"):
            # Function for mixture ensemble
            # Function on main level to allow for parallelization

            # Simulate ensemble for mixture
            pred_agg["f"] = np.asarray(
                list(
                    map(
                        partial(
                            fn_apply_bqn,
                            **dict(kwargs, n_ens=n_ens, f_ls=f_ls),
                        ),
                        range(len(y_test)),
                    )
                )
            )
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     pred_agg["f"] = np.asarray(
            #         executor.map(
            #             partial(
            #                 fn_apply_bqn,
            #                 **dict(kwargs, n_ens=n_ens, f_ls=f_ls),
            #             ),
            #             range(len(y_test)),
            #         )
            #     )

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                rpy_elements=rpy_elements,
            )

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_lp_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan
        elif (temp_nn == "bqn") & (temp_agg == "vi"):
            # Average parameters
            pred_agg["alpha"] = alpha_sum / n_ens
            pred_agg["f"] = f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = (
                np.mean(pred_agg["alpha"], axis=1) - y_test
            )

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_q_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan
        elif (temp_nn == "bqn") & (temp_agg == "vi-w"):
            # Wrapper function
            def fn_optim_bqn_vi_w(x):
                return fn_vi_bqn(
                    a=0,
                    w=x,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = (
                np.mean(pred_agg["alpha"], axis=1) - y_test
            )

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_q_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan
        elif (temp_nn == "bqn") & (temp_agg == "vi-a"):
            # Wrapper function
            def fn_optim_bqn_vi_a(x):
                return fn_vi_bqn(
                    a=x,
                    w=1 / n_ens,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Optimize
            est = minimize(fun=fn_optim_bqn_vi_a, x0=0, method="Nelder-Mead")

            # Read out intercept
            pred_agg["a"] = est.x

            # Optimally weighted parameters
            pred_agg["alpha"] = alpha_sum / n_ens
            pred_agg["f"] = pred_agg["a"] + f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = (
                pred_agg["a"] + np.mean(pred_agg["alpha"], axis=1)
            ) - y_test

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_q_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan
        elif (temp_nn == "bqn") & (temp_agg == "vi-aw"):
            # Wrapper function
            def fn_optim_bqn_vi_aw(x):
                return fn_vi_bqn(
                    a=x[0],
                    w=x[1],
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["a"] + pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = (
                pred_agg["a"] + np.mean(pred_agg["alpha"], axis=1) - y_test
            )

            # Transform ranks to PIT
            pred_agg["scores"]["pit"] = fn_upit(
                ranks=pred_agg["scores"]["rank"],
                max_rank=(kwargs["n_q_samples"] + 1),
            )

            # No ranks
            pred_agg["scores"]["rank"] = np.nan

        # Take time
        end_time = time_ns()

        # Name of file
        filename = (
            f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
        )
        temp_data_out_path = os.path.join(kwargs["data_out_path"], filename)
        # Save aggregated forecasts and scores
        with open(temp_data_out_path, "wb") as f:
            pickle.dump(pred_agg, f)

        print(
            f"{temp_nn.upper()}, {dataset.upper()}:",
            f"Finished aggregation of {filename} -",
            f"{(end_time - start_time)/1e+9:.2f}s",
        )
        # Delete and clean
        del pred_agg

    # Delete
    if temp_nn == "drn":
        del f_sum, f_valid_sum
    elif temp_nn == "bqn":
        del f_sum, alpha_sum, f_valid_sum
    elif temp_nn == "hen":
        pass


def main():
    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    ### Initialize ###
    # Cores to use
    num_cores = CONFIG["NUM_CORES"]

    # Network variantes
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # Aggregation methods
    agg_meths_ls = CONFIG["PARAMS"]["AGG_METHS_LS"]

    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulations
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Size of network ensembles
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Ensemble sizes to be combined
    step_size = 2
    n_ens_vec = np.arange(
        start=step_size, stop=n_ens + step_size, step=step_size
    )

    # Size of LP mixture samples
    # n_lp_samples = 100
    n_lp_samples = 1_000  # 1000 solves problem of relatively low LP CRPSS

    # Size of BQN quantile samples
    n_q_samples = 100

    # Quantile levels for evaluation
    q_levels = np.arange(
        start=1 / (n_q_samples + 1), stop=1, step=1 / (n_q_samples + 1)
    )

    ### Initialize parallel computing ###
    # Grid for parallel computing
    run_grid = pd.DataFrame(columns=["dataset", "temp_nn", "n_ens", "i_sim"])
    for dataset in dataset_ls:
        data_in_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["RESULTS_DIR"],
            dataset,
            CONFIG["ENS_METHOD"],
            CONFIG["PATHS"]["ENSEMBLE_F"],
        )
        data_out_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["RESULTS_DIR"],
            dataset,
            CONFIG["ENS_METHOD"],
            CONFIG["PATHS"]["AGG_F"],
        )
        if dataset.startswith("scen"):
            temp_n_sim = n_sim
        else:
            temp_n_sim = 20
        for temp_nn in nn_vec:
            for i_ens in n_ens_vec[::-1]:
                for i_sim in range(temp_n_sim):
                    new_row = {
                        "dataset": dataset,
                        "temp_nn": temp_nn,
                        "n_ens": i_ens,
                        "i_sim": i_sim,
                        "data_in_path": data_in_path,
                        "data_out_path": data_out_path,
                    }

                    run_grid = pd.concat(
                        [run_grid, pd.DataFrame(new_row, index=[0])],
                        ignore_index=True,
                    )

    ### Parallel-Loop ###
    # Maximum number of cores
    num_cores = min(num_cores, run_grid.shape[0])

    # Take time
    total_start_time = time_ns()

    # Run sequential or run parallel
    run_parallel = True

    print(run_grid.shape[0])
    if run_parallel:
        ### Run parallel ###
        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(fn_mc)(
                temp_nn=row["temp_nn"],
                dataset=row["dataset"],
                n_ens=row["n_ens"],
                i_sim=row["i_sim"],
                q_levels=q_levels,
                nn_vec=nn_vec,
                agg_meths_ls=agg_meths_ls,
                data_in_path=row["data_in_path"],
                data_out_path=row["data_out_path"],
                n_lp_samples=n_lp_samples,
                n_q_samples=n_q_samples,
                num_cores=int(num_cores / 2),
            )
            for _, row in run_grid.iterrows()
        )
    else:
        ### Run sequential ###
        for _, row in run_grid.iterrows():
            fn_mc(
                temp_nn=row["temp_nn"],
                dataset=row["dataset"],
                n_ens=row["n_ens"],
                i_sim=row["i_sim"],
                q_levels=q_levels,
                nn_vec=nn_vec,
                agg_meths_ls=agg_meths_ls,
                data_in_path=row["data_in_path"],
                data_out_path=row["data_out_path"],
                n_lp_samples=n_lp_samples,
                n_q_samples=n_q_samples,
                num_cores=num_cores,
            )

    # Take time
    total_end_time = time_ns()

    # Print processing time
    print(
        "Finished processing of all threads within",
        f"{(total_end_time - total_start_time) / 1e+9:.2f}s",
    )


if __name__ == "__main__":
    main()
