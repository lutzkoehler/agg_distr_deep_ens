## Simulation study: Script 2
# Aggregation of deep ensembles

import os
import pickle
from itertools import product
from functools import partial
from random import choices
from time import time_ns

import numpy as np
import pandas as pd
import scipy.stats as ss
from joblib import Parallel, delayed
from multiprocessing.pool import Pool
from rpy2.robjects import default_converter, numpy2ri, vectors
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.optimize import minimize

from fn_basic import fn_upit
from fn_eval import bern_quants, fn_scores_distr, fn_scores_ens


### Functions for weight estimation (and uPIT) ###
def fn_vi_drn(a, w, f_sum, y, **kwargs):
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
    _type_
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

    # Calculate quantiles
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
    i_rep = choices(
        population=range(n_ens), k=kwargs["n_lp_samples"]
    )  # with replacement

    # Draw from individual distributions
    # TODO: Check if works
    # def func_temp(j):
    #    alpha_temp = kwargs["f_ls"][f"alpha{j}"][i, :]

    # return bern_quants(
    #    q_levels=np.random.uniform(size=1), alpha=alpha_temp
    # )

    alpha_vec = [
        bern_quants(
            alpha=kwargs["f_ls"][f"alpha{j}"][i, :],
            q_levels=np.random.uniform(size=1),
        )
        for j in i_rep
    ]

    # res = np.apply_along_axis(
    #    func1d=bern_quants(
    #        alpha=alpha_temp[j],
    #        q_levels=np.random.uniform(size=1),
    #    ),
    #    axis=0,
    #    arr=i_rep,
    #    q_levels=np.random.uniform(size=1),
    # )

    return np.reshape(np.asarray(alpha_vec), newshape=(kwargs["n_lp_samples"]))


### Parallel-Function ###
# Function for parallel computing
def fn_mc(i_nn, i_scenario, n_ens, i_sim, **kwargs):
    ### Initialization ###
    # Create pool for parallel processing
    current_pool = Pool()

    # Read out network type
    temp_nn = kwargs["nn_vec"][i_nn]

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
        filename = f"{temp_nn}_scen_{i_scenario}_sim_{i_sim}_ens_{i_ens}.pkl"
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
            f_valid_ls[f"alpha{i_ens}"] = pred_nn["f"][i_valid, :]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
            f_ls[f"alpha{i_ens}"] = pred_nn["f"][i_test, :]
        elif temp_nn == "hen":
            pass

    # Average forecasts depending on method
    f_sum = []
    f_valid_sum = []
    alpha_sum = []
    if temp_nn == "drn":
        # Average parameters
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
                i_rep = choices(
                    population=range(n_ens), k=kwargs["n_lp_samples"]
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
            pred_agg["scores"] = fn_scores_ens(ens=pred_agg["f"], y=y_test)

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
            pred_agg["w"] = est["x"]

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
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
            pred_agg["a"] = est["x"]

            # Calculate equally weighted VI
            pred_agg["f"] = f_sum / n_ens

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr="norm",
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
            pred_agg["a"] = est["x"][0]
            pred_agg["w"] = est["x"][1]

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"], y=y_test, distr="norm"
            )
        elif (temp_nn == "bqn") & (temp_agg == "lp"):
            # Function for mixture ensemble
            # Function on main level to allow for parallelization

            # Simulate ensemble for mixture
            # pred_agg["f"] = np.asarray(
            #    map(
            #        lambda x: fn_apply_bqn(
            #            x, **dict(kwargs, n_ens=n_ens, f_ls=f_ls)
            #        ),
            #        range(len(y_test)),
            #    )
            # )

            pred_agg["f"] = np.asarray(
                current_pool.map(
                    func=partial(
                        fn_apply_bqn, **dict(kwargs, n_ens=n_ens, f_ls=f_ls)
                    ),
                    iterable=range(len(y_test)),
                )
            )

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_ens(ens=pred_agg["f"], y=y_test)

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
            pred_agg["w"] = est["x"]

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
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
            pred_agg["a"] = est["x"]

            # Optimally weighted parameters
            pred_agg["alpha"] = alpha_sum / n_ens
            pred_agg["f"] = pred_agg["a"] + f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
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
            pred_agg["a"] = est["x"][0]
            pred_agg["w"] = est["x"][1]

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["a"] + pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
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
        filename = f"{temp_nn}_scen_{i_scenario}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
        temp_data_out_path = os.path.join(kwargs["data_out_path"], filename)
        # Save aggregated forecasts and scores
        with open(temp_data_out_path, "wb") as f:
            pickle.dump(pred_agg, f)

        print(
            f"{temp_nn.upper()}: Finished aggregation of {filename}"
            f" - {(end_time - start_time)/1e+9}s",
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
    ### Settings ###
    # Path of deep ensemble forecasts
    data_in_path = os.path.join("data", "model")

    # Path of aggregated forecasts
    data_out_path = os.path.join("data", "agg")

    ### Initialize ###
    # Cores to use
    num_cores = 7

    # Network variantes
    nn_vec = ["drn", "bqn"]  # For now without "hen"
    # nn_vec = ["bqn"]

    # Aggregation methods
    # lp -> Linear pool
    # vi -> Vincentization
    # vi-w -> Vincentization with weight estimation
    # vi-a -> Vincentization with intercept estimation
    # vi-aw -> Vincentization with weight and intercept estimation
    agg_meths_ls = {
        "drn": ["lp", "vi", "vi-w", "vi-a", "vi-aw"],
        "bqn": ["lp", "vi", "vi-w", "vi-a", "vi-aw"],
    }

    # Models considered
    # scenario_vec = range(1, 7, 1)
    scenario_vec = [1, 4]

    # Number of simulations
    # n_sim = 50
    n_sim = 10

    # Ensemble sizes to be combined
    n_ens_vec = np.arange(start=2, stop=12, step=2)

    # Size of LP mixture samples
    n_lp_samples = 100

    # Size of BQN quantile samples
    n_q_samples = 100

    # Quantile levels for evaluation
    q_levels = np.arange(
        start=1 / (n_q_samples + 1), stop=1, step=1 / (n_q_samples + 1)
    )

    ### Initialize parallel computing ###
    # Grid for parallel computing
    grid_par = pd.DataFrame(
        list(
            product(
                range(len(nn_vec)),
                scenario_vec,
                np.sort(n_ens_vec)[::-1],
                range(n_sim),
            )
        ),
        columns=["nn_vec", "scenario_vec", "n_ens_vec", "n_sim"],
    )

    ### Parallel-Loop ###
    # Maximum number of cores
    num_cores = min(num_cores, grid_par.shape[0])

    # Take time
    total_start_time = time_ns()

    ### Run sequential ###
    print(grid_par.shape[0])
    for _, row in grid_par.iterrows():
        fn_mc(
            i_nn=row["nn_vec"],
            i_scenario=row["scenario_vec"],
            n_ens=row["n_ens_vec"],
            i_sim=row["n_sim"],
            q_levels=q_levels,
            nn_vec=nn_vec,
            agg_meths_ls=agg_meths_ls,
            data_in_path=data_in_path,
            data_out_path=data_out_path,
            n_lp_samples=n_lp_samples,
            n_q_samples=n_q_samples,
        )

    ### Run parallel ###
    # Parallel(n_jobs=num_cores, backend="multiprocessing")(
    #    delayed(fn_mc)(
    #        i_nn=row["nn_vec"],
    #        i_scenario=row["scenario_vec"],
    #        n_ens=row["n_ens_vec"],
    #        i_sim=row["n_sim"],
    #        q_levels=q_levels,
    #        nn_vec=nn_vec,
    #        agg_meths_ls=agg_meths_ls,
    #        data_in_path=data_in_path,
    #        data_out_path=data_out_path,
    #        n_lp_samples=n_lp_samples,
    #        n_q_samples=n_q_samples,
    #    )
    #    for _, row in grid_par.iterrows()
    # )

    # Take time
    total_end_time = time_ns()

    # Print processing time
    print(
        f"Finished processing of all threads"
        f"within {(total_end_time - total_start_time) / 1e+9}s"
    )


if __name__ == "__main__":
    main()
