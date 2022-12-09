## Simulation study: Script 0
# Simulate underlying data

import os
import pickle
import itertools

import numpy as np
from typing import Any
from nptyping import NDArray, Float, Int
import scipy.stats as ss
from joblib import Parallel, delayed
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from fn_eval import fn_scores_distr, fn_scores_ens


def simulate_data(i_scenario: int, i_sim: int, data_out_path: str) -> None:
    ### Setup R function calls ###
    # Initialize rpy elements for all scoring functions
    multiRNG = importr("MultiRNG")
    np_cv_rules = default_converter + numpy2ri.converter
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": np_cv_rules,
    }

    ### Initiation ###
    # Set seed
    np.random.seed(123 * i_sim)

    # Size of training sets
    if i_scenario == 6:
        n_train: int = int(3000)
    else:
        n_train: int = int(6000)

    # Size of test sets
    n_test: int = int(1e4)

    # Indices for training and test set
    i_train: list = list(range(n_train))
    i_test: list = [x + n_train for x in range(n_test)]

    # Number of predictors for each model
    if i_scenario == 3:
        n_preds: int = int(1)
    else:
        n_preds: int = int(5)

    ### Generate data ###
    # Initialize empty X, y, beta_1, beta_2, bn
    X: NDArray[Any, Float] = np.empty(shape=(n_train + n_test, n_preds))
    y: NDArray[Any, Float] = np.empty(shape=(n_train + n_test))
    beta_1: NDArray[Any, Float] = np.empty(shape=(n_preds))
    beta_2: NDArray[Any, Float] = np.empty(shape=(n_preds))
    bn: NDArray[Any, Int] = np.empty(shape=(n_train + n_test), dtype=np.int64)

    # Differentiate models
    if i_scenario == 1:
        # Coefficients
        beta_1 = np.random.normal(loc=0, scale=1, size=n_preds)
        beta_2 = np.random.normal(loc=0, scale=0.45, size=n_preds)

        # Predictors
        X = np.reshape(
            a=np.random.normal(
                loc=0, scale=1, size=(n_train + n_test) * n_preds
            ),
            newshape=(n_train + n_test, n_preds),
        )

        # Observational error
        eps = np.random.normal(loc=0, scale=1, size=n_train + n_test)

        # Calculate observations
        y = np.add(np.dot(X, beta_1), np.exp(np.dot(X, beta_2)) * eps)

    elif i_scenario in [2, 5, 6]:
        # Predictors
        if i_scenario == 5:
            # Generate covariance matrix
            sigma = np.ones(shape=(n_preds, n_preds))

            # For-Loop over rows and columns
            for i, j in np.ndindex(sigma.shape):
                sigma[i, j] = 0.5 ** abs(i - j)

            # Draw correlated variables
            with localconverter(np_cv_rules) as cv:  # noqa: F841
                X = multiRNG.draw_d_variate_uniform(
                    no_row=n_train + n_test, d=n_preds, cov_mat=sigma
                )

        else:
            # Draw iid uniform random variables
            X = np.reshape(
                np.random.uniform(
                    low=0, high=1, size=(n_train + n_test) * n_preds
                ),
                newshape=(n_train + n_test, n_preds),
            )

        # Bernoulli variable
        bn = np.random.binomial(n=1, p=0.5, size=n_train + n_test)

        # Observational error
        eps_1 = np.random.normal(
            loc=0, scale=1.5, size=n_train + n_test
        )  # var = 2.25
        eps_2 = np.random.normal(loc=0, scale=1, size=n_train + n_test)

        # Calculate observations
        y = bn * (
            10 * np.sin(2 * np.pi * X[:, 0] * X[:, 1]) + 10 * X[:, 3] + eps_1
        ) + (1 - bn) * (20 * (X[:, 2] - 0.5) ** 2 + 5 * X[:, 4] + eps_2)

    elif i_scenario == 3:
        # Predictors
        X = np.reshape(
            np.random.uniform(
                low=0, high=10, size=(n_train + n_test) * n_preds
            ),
            newshape=(n_train + n_test, n_preds),
        )

        # Bernoulli variable
        bn = np.random.binomial(n=1, p=0.5, size=n_train + n_test)

        # Observational error
        eps_1 = np.random.normal(
            loc=0, scale=0.3, size=n_train + n_test
        )  # var = 0.09
        eps_2 = np.random.normal(
            loc=0, scale=0.8, size=n_train + n_test
        )  # var = 0.64

        # Calculate observations
        y = bn * (np.sin(X[:, 0]) + eps_1) + (1 - bn) * (
            2 * np.sin(1.5 * X[:, 0] + 1) + eps_2
        )

    elif i_scenario == 4:
        # Predictors
        X = np.reshape(
            np.random.uniform(
                low=0, high=1, size=(n_train + n_test) * n_preds
            ),
            newshape=(n_train + n_test, n_preds),
        )

        # Observational error
        eps = ss.skewnorm.rvs(
            loc=0,  # xi
            scale=1,  # omega
            a=-5,  # alpha
            size=n_train + n_test,
        )

        # Calculate observations
        y = (
            10 * np.sin(2 * np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
            + eps
        )

    ### Data partition ###
    # Split in training and testing
    X_train: NDArray[Any, Float] = X[i_train, :]  # type: ignore
    X_test: NDArray[Any, Float] = X[i_test, :]  # type: ignore
    y_train: NDArray[Any, Float] = y[i_train]
    y_test: NDArray[Any, Float] = y[i_test]

    ### Optimal forecast ###
    # Generate matrix for parameter forecasts / sample
    # Initialize f_opt as empty
    f_opt = np.empty(shape=(len(y_test), 2))
    n_sample: int = 0  # To get rid of warnings
    if i_scenario == 4:
        # Number of samples to draw
        n_sample = 1000
    else:
        # Normal distribution
        # Col 0: loc, col 1: scale
        f_opt = np.empty(shape=(len(y_test), 2))

    # Differentiate scenarios
    if i_scenario == 1:
        # Location parameter
        f_opt[:, 0] = np.dot(X_test, beta_1)

        # Scale parameter (standard deviation)
        f_opt[:, 1] = np.exp(np.dot(X_test, beta_2))

    elif i_scenario in [2, 5, 6]:
        ## Assumption Bernoulli variable is known (elsewise multimodal)
        # Location
        f_opt[:, 0] = bn[i_test] * (
            10 * np.sin(2 * np.pi * X_test[:, 0] * X_test[:, 1])
            + 10 * X_test[:, 3]
        ) + (1 - bn[i_test]) * (
            20 * (X_test[:, 2] - 0.5) ** 2 + 5 * X_test[:, 4]
        )

        # Scale parameter (standard deviation)
        f_opt[:, 1] = bn[i_test] * 1.5 + (1 - bn[i_test]) * 1

    elif i_scenario == 3:
        ## Assumption Bernoulli variable is known (elsewise multimodal)
        # Location
        f_opt[:, 0] = bn[i_test] * (np.sin(X_test[:, 0])) + (
            1 - bn[i_test]
        ) * (2 * np.sin(1.5 * X_test[:, 0] + 1))

        # Scale parameter (standard deviation)
        f_opt[:, 1] = bn[i_test] * 0.3 + (1 - bn[i_test]) * 0.8

    elif i_scenario == 4:
        # Draw samples from a skewed normal
        f_opt = np.apply_along_axis(
            func1d=lambda x: ss.skewnorm.rvs(
                loc=(
                    10 * np.sin(2 * np.pi * x[0] * x[1])
                    + 20 * (x[2] - 0.5) ** 2
                    + 10 * x[3]
                    + 5 * x[4]
                ),  # xi
                scale=1,  # omega
                a=-5,  # alpha
                size=n_sample,
            ),
            axis=1,
            arr=X_test,
        )

    # Optimal scores
    if i_scenario == 4:
        # Number of samples to draw
        scores_opt = fn_scores_ens(
            ens=f_opt, y=y_test, rpy_elements=rpy_elements
        )

        # Transform ranks to uPIT
        scores_opt["pit"] = scores_opt["rank"] / (
            n_sample + 1
        ) - np.random.uniform(low=0, high=1 / (n_sample + 1), size=n_test)

        # Omit ranks
        scores_opt["rank"] = np.nan

    else:
        # Normal distribution
        scores_opt = fn_scores_distr(
            f=f_opt, y=y_test, distr="norm", rpy_elements=rpy_elements
        )

    ### Save data ###
    # Save ensemble member
    save_path = os.path.join(
        data_out_path, f"scen_{i_scenario}_sim_{i_sim}.pkl"
    )
    with open(save_path, "wb") as f:
        pickle.dump([X_train, y_train, X_test, y_test, f_opt, scores_opt], f)
    print(f"Finished run {i_sim} of scenario {i_scenario}")


if __name__ == "__main__":
    ### Settings ###
    # Path for simulated data
    data_out_path = os.path.join("data", "ss_data")

    ### Initialize ###
    # Models considered
    # scenario_vec = range(1, 7, 1)
    scenario_vec = [1, 4]

    # Number of simulations
    # n_sim = 50
    n_sim: int = 10

    ### Run sequential ###
    # for i_scenario in scenario_vec:
    #    for i_sim in range(n_sim):
    #        simulate_data(i_scenario=i_scenario, i_sim=i_sim)

    ### Run parallel ###
    Parallel(n_jobs=7, backend="multiprocessing")(
        delayed(simulate_data)(
            i_scenario=i_scenario, i_sim=i_sim, data_out_path=data_out_path
        )
        for i_scenario, i_sim in itertools.product(scenario_vec, range(n_sim))
    )
