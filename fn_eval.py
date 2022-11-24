## Function file
# Evaluation of probabilistic forecasts

import numpy as np
import pandas as pd
import scipy.stats as ss
from rpy2.robjects import default_converter, numpy2ri, vectors
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


### Coverage ###
def fn_cover(x, alpha=None, n_ens=20):
    """Calculate coverage of a probabilistic forecast

    Parameters
    ----------
    x : n vector
        PIT values / ranks
    alpha : ((n_ens - 1)/(n_ens + 1)), optional
        Significance level (probability), by default None
        -> Nominal n_ens-member coverage
    n_ens : int, optional
        Size of ensemble, by default 20 -> COSMO-DE-EPS

    Returns
    -------
    n vector
        Coverage in percentage
    """
    ### Initiation ###
    # Nominal COSMO coverage
    if alpha is None:
        alpha = 2 / (n_ens + 1)

    ### Coverage calculation ###
    # PIT or rank?
    if min(x) < 1:
        res = np.mean(x.loc[(alpha / 2 <= x) & (x <= (1 - alpha / 2))])
    else:
        res = np.mean(x[x in range(2, n_ens + 1)])

    # Output as percentage
    return 100 * res


### BQN: Bernstein Quantile function ###
def bern_quants(alpha, q_levels):
    """Function that calculates quantiles for given coefficients

    Parameters
    ----------
    alpha : n x (p_degree + 1) matrix
        Coefficients of Bernstein basis
    q_levels : n_q vector
        Quantile levels

    Returns
    -------
    n x n_q matrix
        Quantile forecasts for given coefficients
    """
    ### Initiation ###
    if len(alpha.shape) == 1:
        p_degree = alpha.shape[0] - 1
    else:
        p_degree = alpha.shape[1] - 1

    ### Calculation ###
    # Calculate quantiles (sum of coefficients times basis polynomials)
    # fac = np.asarray(
    #    list(
    #        map(
    #            lambda x: ss.binom.pmf(x, n=p_degree, p=q_levels),
    #            range(p_degree + 1),
    #        )
    #    )
    # )
    if len(q_levels) == 1:
        fac = ss.binom.pmf(list(range(p_degree + 1)), n=p_degree, p=q_levels)
        return np.dot(alpha, fac)
    else:
        fac = [
            ss.binom.pmf(list(range(p_degree + 1)), n=p_degree, p=current_q)
            for current_q in q_levels
        ]
        fac = np.asarray(fac)
        return np.dot(alpha, fac.T)

    # if len(q_levels) == 1:
    #    res = np.dot(alpha, fac)
    # else:
    #    res = np.dot(alpha, fac.T)

    # Return quantiles
    # return res


### ENS: Evaluation of ensemble ###
def fn_scores_ens(ens, y, skip_evals=None, scores_ens=True):
    """Function to calculate evaluation measures of scores

    Parameters
    ----------
    ens : n x n_ens matrix
        Ensemble data for prediction
    y : n vector
        Observations for prediction
    skip_evals : string vector
        Skip the following evaluation measures, Default: None -> Calculate
        all
    scores_ens : bool
        Should scores of ensemble forecasts, interval lengths, ranks be
        calculated?, Default: True -> Calculate

    Returns
    -------
    scores_ens : n x 6 DataFrame
        DataFrame containing:
            rank : n vector
                Ranks of observations in ensemble forecasts
            crps : n vector
                CRPS of ensemble forecasts
            logs : n vector
                Log-Score of ensemble forecasts
            lgt : n vector
                Ensemble range
            e_md : n vector
                Bias of median forecast
            e_me : n vector
                Bias of mean forecast

    Raises
    ------
    Exception
        If ensemble should not be scored
    """
    ### Initiation ###
    # Load R packages
    # To use np arrays in rpy2:
    base = importr("base")
    scoring_rules = importr("scoringRules")
    np_cv_rules = default_converter + numpy2ri.converter
    # Convert y to rpy2.robjects.vectors.FloatVector
    y_vector = vectors.FloatVector(y)

    # Calculate only if scores_ens is True
    if not scores_ens:
        raise Exception("Ensemble should not be scored.")

    # Size of COSMO-DE-EPS
    n_cosmo = 20

    # Get number of ensembles
    n = ens.shape[0]

    # Get ensemble size
    n_ens = ens.shape[1]

    # Make data frame
    scores_ens = pd.DataFrame(
        columns=["rank", "crps", "logs", "lgt", "e_me", "e_md"], index=range(n)
    )

    ### Calculation ###
    # Calculate observation ranks
    if "rank" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:
            scores_ens["rank"] = np.apply_along_axis(
                func1d=lambda x: base.rank(x, ties="random")[0],
                axis=1,
                arr=np.c_[y, ens],
            )

    # Calculate CRPS or raw ensemble
    if "crps" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:
            scores_ens["crps"] = scoring_rules.crps_sample(y=y_vector, dat=ens)

    # Calculate Log-Score of raw ensemble
    if "logs" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:  # noqa: F841
            scores_ens["logs"] = scoring_rules.logs_sample(y=y_vector, dat=ens)

    # Calculate ~(20 - 1)/(20 + 1)% prediction interval (corresponds to COSMO
    # ensemble range)
    if "lgt" in scores_ens.columns:
        # COSMO-ensemble size
        if n_ens == n_cosmo:
            scores_ens["lgt"] = np.apply_along_axis(
                func1d=np.ptp,
                axis=1,
                arr=ens,
            )
        # Corresponding quantiles (1/21 and 20/21) are included
        elif (n_ens + 1) % (n_cosmo + 1) == 0:
            # TODO: Verify if works correct
            # Indices of corresponding quantiles
            i_lgt = (n_ens + 1) / (n_cosmo + 1) * np.hstack((1, n_cosmo))

            # Get quantiles
            q_lgt = np.sort(a=ens, axis=1).T[:, i_lgt]

            # Transform in vector
            # TODO: Necessary?

            # Calculate corresponding range
            scores_ens["lgt"] = np.apply_along_axis(
                func1d=np.ptp,
                axis=1,
                arr=q_lgt,
            )
        # Quantiles are not included: Calculate corresponding via quantile
        # function
        else:
            # Choose type 8, as suggested in ?quantile (and bias observed for
            # LP for default)
            # TODO: look for method="median_unbiased" quantile
            scores_ens["lgt"] = np.apply_along_axis(
                func1d=lambda x: np.diff(
                    np.quantile(
                        a=x,
                        q=np.hstack((1, 20)) / (n_cosmo + 1),
                        # method="median_unbiased",
                    )
                ),
                axis=1,
                arr=ens,
            )

    # Calculate bias of mean forecast
    if "e_me" in scores_ens.columns:
        scores_ens["e_me"] = np.mean(a=ens, axis=1) - y

    # Calculate bias of median forecast
    if "e_md" in scores_ens.columns:
        scores_ens["e_md"] = np.median(a=ens, axis=1) - y

    ### Output ###
    # Skip evaluation measures
    if skip_evals is not None:
        scores_ens = scores_ens.drop(columns=skip_evals)

    # Return output
    return scores_ens


def fn_scores_distr(f, y, distr="tlogis", n_ens=20, skip_evals=None):
    """Function for prediciton based on the distributional parameters

    Parameters
    ----------
    f : n x n_par matrix
        Ensemble data for prediction
    y : n vector
        Observations
    distr : "tlogis", "norm"
        Parametric distribution, Default: (zero-)truncated logistic
    n_ens : int
        Ensemble size: Used for confidence level of prediction intervals
    skip_evals : string vector
        Skip the following evaluation measures, Default: None -> Calculate
        all

    Returns
    -------
    scores_pp : n x 6 DataFrame
        DataFrame containing:
            pit : n vector
                PIT values of distributional forecasts
            crps : n vector
                CRPS of forecasts
            logs : n vector
                Log-Score of forecasts
            lgt : n vector
                Length of prediction interval
            e_md : n vector
                Bias of median forecast
            e_me : n vector
                Bias of mean forecast
    """
    ### Initiation ###
    # Load R packages
    scoring_rules = importr("scoringRules")
    crch = importr("crch")
    np_cv_rules = default_converter + numpy2ri.converter
    # Convert y to rpy2.robjects.vectors.FloatVector
    y_vector = vectors.FloatVector(y)

    # Input check
    if distr not in ["tlogis", "norm"] and any(f[:, 1] < 0):
        print("Non-positive scale forecast!")

    ### Data preparation ###
    # Number of predictions
    n = f.shape[0]

    # Make data frame
    scores_pp = pd.DataFrame(
        index=range(n), columns=["pit", "crps", "logs", "lgt", "e_me", "e_md"]
    )

    ### Prediction and score calculation ###
    # Forecasts depending on distribution
    if distr == "tlogis":  # truncated logistic
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["pit"] = crch.ptlogis(
                    q=y, location=f[:, 0], scale=f[:, 1], left=0
                )

        # Calculate CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_tlogis(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["logs"] = scoring_rules.logs_tlogis(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
        if "lgt" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["lgt"] = crch.qtlogis(
                    p=n_ens / (n_ens + 1),
                    location=f[:, 0],
                    scale=f[:, 1],
                    left=0,
                ) - crch.qtlogis(
                    p=1 / (n_ens + 1),
                    location=f[:, 0],
                    scale=f[:, 1],
                    left=0,
                )

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["e_md"] = (
                    crch.qtlogis(
                        p=0.5,
                        location=f[:, 0],
                        scale=f[:, 1],
                        left=0,
                    )
                    - y
                )

        # Calculate bias of mean forecast
        if "e_me" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["e_me"] = (
                    f[:, 1]
                    - f[:, 2] * np.log(1 - crch.plogis(-f[:, 0] / f[:, 1]))
                ) / (1 - crch.plogis(-f[:, 0] / f[:, 1])) - y

    elif distr == "norm":  # normal
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            scores_pp["pit"] = ss.norm.cdf(x=y, loc=f[:, 0], scale=f[:, 1])

        # Calculate CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_norm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1]
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:  # noqa: F841
                scores_pp["logs"] = scoring_rules.logs_norm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1]
                )

        # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
        if "lgt" in scores_pp.columns:
            scores_pp["lgt"] = ss.norm.ppf(
                q=n_ens / (n_ens + 1), loc=f[:, 0], scale=f[:, 1]
            ) - ss.norm.ppf(q=1 / (n_ens + 1), loc=f[:, 0], scale=f[:, 1])

        # Calculate bias of mean forecast
        scores_pp["e_me"] = f[:, 0] - y

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            scores_pp["e_md"] = (
                ss.norm.ppf(q=0.5, loc=f[:, 0], scale=f[:, 1]) - y
            )

        ### Output ###
        # Skip evaluation measures
        if skip_evals is not None:
            scores_pp = scores_pp.drop(columns=skip_evals)

        # Return
        return scores_pp
