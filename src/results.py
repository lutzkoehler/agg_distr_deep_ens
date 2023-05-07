import os
import pickle

import numpy as np
import pandas as pd


def get_panel_data(
    data_path: str,
    dataset_ls: list[str],
    score_vec: list[str],
    nn_vec: list[str],
    n_ens_vec: list[int],
    agg_meths: list[str],
    ens_method: str,
) -> pd.DataFrame:
    """Creates DataFrame for the list of
    [datasets, NN types, ensemble sizes, agg methods] and one ens_method.

    Parameters
    ----------
    data_path : string
        Path to eval_{dataset}_{ens_method}.pkl files
    dataset_ls : list[string]
        List containing the names of datasets to summarize
    score_vec : list[string]
        List containing the names of scores to summarize
        (e.g. crps, a, w, etc.)
    nn_vec : list[string]
        List containing the NN types to consider ("drn", "bqn")
    n_ens_vec : list[int]
        List containing ensemble sizes to consider
    agg_meths : list[string]
        List containing the aggregation methods to consider (e.g. "lp", "vi")
    ens_method : string
        Ensemble method name

    Returns
    -------
    DataFrame
        Columns: dataset, nn, metric, n_ens, agg, score
    """
    df_plot = pd.DataFrame()
    # For-Loop over scenarios
    for dataset in dataset_ls:
        ### Simulation: Load data ###
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_path = data_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_path, filename), "rb") as f:
            df_scores = pickle.load(f)

        # Check dataset is simulated or UCI Dataset
        if dataset.startswith("scen"):
            optimal_scores_available = True
        else:
            optimal_scores_available = False

        ### Initialization ###
        # Only scenario
        df_sc = df_scores[df_scores["model"] == dataset]

        ### Calculate quantities ###

        # For-Loop over quantities of interest
        for temp_sr in score_vec:
            # Consider special case CRPSS
            if temp_sr == "crpss":
                temp_out = "crps"
            else:
                temp_out = temp_sr

            # Get optimal score of scenario for CRPSS
            s_opt = 0
            if optimal_scores_available and (temp_sr not in ["a", "w"]):
                s_opt = df_sc[df_sc["type"] == "ref"][temp_out].mean()

            # For-Loop over network variants
            for temp_nn in nn_vec:
                # Only network type
                df_nn = df_sc[df_sc["nn"] == temp_nn]

                # For-Lop over ensemble sizes and aggregation methods
                for i_ens in n_ens_vec:
                    for temp_agg in ["opt", "ens"] + agg_meths:
                        # Skip ensemble for skill
                        if (temp_sr == "crpss") and (temp_agg == "ens"):
                            continue
                        elif (temp_sr == "crpss") and (temp_agg == "opt"):
                            continue
                        elif (temp_sr == "a") and (temp_agg == "opt"):
                            continue
                        elif (temp_sr == "w") and (temp_agg == "opt"):
                            continue

                        # Fill in data frame
                        new_row = {
                            "dataset": dataset,
                            "nn": temp_nn,
                            "metric": temp_sr,
                            "n_ens": i_ens,
                            "agg": temp_agg,
                        }

                        # Reference: Average score of ensemble members
                        s_ref = df_nn[
                            (df_nn["n_rep"] <= i_ens)
                            & (df_nn["type"] == "ind")
                        ][temp_out].mean()

                        # Special case: Average ensemble score
                        if temp_agg == "ens":
                            new_row["score"] = s_ref
                        elif temp_agg == "opt":
                            new_row["score"] = s_opt
                        else:
                            # Read out score
                            new_row["score"] = df_nn[
                                (df_nn["n_ens"] == i_ens)
                                & (df_nn["type"] == temp_agg)
                            ][temp_out].mean()

                            # Special case: CRPSS
                            if temp_sr == "crpss":
                                # Calcuate skill
                                new_row["score"] = (
                                    100
                                    * (s_ref - new_row["score"])
                                    / (s_ref - s_opt)
                                )

                            # Relative weight difference to equal weights in %
                            if temp_sr == "w":
                                new_row["score"] = 100 * (
                                    i_ens * new_row["score"] - 1
                                )

                        df_plot = pd.concat(
                            [
                                df_plot,
                                pd.DataFrame(new_row, index=[0]),
                            ],
                            ignore_index=True,
                        )

    return df_plot


def get_best_scores_table(
    data_path: str,
    nn_vec: list[str],
    ens_method_ls: list[str],
    agg_meths: list[str],
    dataset_ls: list[str],
    n_ens_vec: list[int],
) -> dict:
    """Summarize best score for each (dataset, ens_method).
    Also contains the corresponding agg and n_ens.

    Parameters
    ----------
    data_path : string
        Path to eval_{dataset}_{ens_method}.pkl files
    nn_vec : list[string]
        List containing the NN types to consider ("drn", "bqn")
    ens_method_ls : list[string]
        List containing the names of ensembles to consider (e.g. "rand_init)
    agg_meths : list[string]
        List containing the aggregation methods to consider (e.g. "lp", "vi")
    dataset_ls : list[string]
        List containing the names of datasets to summarize
    n_ens_vec : list[int]
        List containing ensemble sizes to consider

    Returns
    -------
    DataFrame
        First entry contains the results for DRNs, second for BQN
    """
    score_vec = ["crps", "crpss", "me", "lgt", "cov", "a", "w"]

    results = {"drn_results": None, "bqn_results": None}

    for ens_method in ens_method_ls:
        # temp_data_path = os.path.join(data_path, ens_method)
        df = get_panel_data(
            data_path,
            dataset_ls,
            score_vec,
            nn_vec,
            n_ens_vec,
            agg_meths,
            ens_method,
        )

        # Basic filtering
        df = df[df["agg"].isin(agg_meths)]
        df = df[df["metric"] == "crps"]
        df["score"] = df["score"].astype("float")

        for nn in nn_vec:
            if nn == "drn":
                temp_results = results["drn_results"]
            else:
                temp_results = results["bqn_results"]
            df_nn = df[df["nn"] == nn]

            if any(df_nn["score"].isna()):
                break

            # Group by and get idx of best scores
            best_row_idx = list(df_nn.groupby("dataset").score.idxmin())
            # Select best row for each dataset
            best_row = df_nn.loc[best_row_idx]

            data = {
                f"{ens_method}": best_row["score"],
                "dataset": best_row["dataset"],
                f"{ens_method}_agg": best_row["agg"],
                f"{ens_method}_n_ens": best_row["n_ens"],
            }
            df_ens_result = pd.DataFrame(data)

            if temp_results is None:
                temp_results = df_ens_result
                temp_results = temp_results.set_index("dataset")
            else:
                df_ens_result = df_ens_result.set_index("dataset")
                temp_results = temp_results.join(df_ens_result)

            if nn == "drn":
                results["drn_results"] = temp_results  # type: ignore
            else:
                results["bqn_results"] = temp_results  # type: ignore

    return results


def get_scores_skills_table(
    data_path: str,
    dataset_ls: list[str],
    score_vec: list[str],
    nn_vec: list[str],
    n_ens_vec: list[int],
    agg_meths: list[str],
    ens_method_ls: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates one scores and one skills DataFrame with all scores / skill for
    each ensemble size in n_ens_vec and for each (ens_method, dataset, nn, agg)

    Parameters
    ----------
    data_path : string
        Path to eval_{dataset}_{ens_method}.pkl files
    dataset_ls : list[string]
        List containing the names of datasets to summarize
    score_vec : list[string]
        List containing the names of scores to summarize
        (e.g. crps, a, w, etc.)
    nn_vec : list[string]
        List containing the NN types to consider ("drn", "bqn")
    n_ens_vec : list[int]
        List containing ensemble sizes to consider
    agg_meths : list[string]
        List containing the aggregation methods to consider (e.g. "lp", "vi")
    ens_method_ls : list[string]
        List containing the names of ensembles to consider (e.g. "rand_init)

    Returns
    -------
    DataFrames
        Contains the skills for each ensemble size
    """
    df_skills = pd.DataFrame()
    df_scores = pd.DataFrame()

    for ens_method in ens_method_ls:
        df = get_panel_data(
            data_path,
            dataset_ls,
            score_vec,
            nn_vec,
            n_ens_vec,
            agg_meths,
            ens_method,
        )
        # df = df_results.dropna()

        for dataset in df["dataset"].unique():
            for nn in df["nn"].unique():
                # Filtered for dataset, NN type and ens_method
                df_temp = df[(df["dataset"] == dataset) & (df["nn"] == nn)]

                for agg in agg_meths + ["ens"]:
                    # Get skill for agg method
                    # Filtered for dataset, NN type, ens_method and
                    # aggregation method
                    df_temp_agg = df_temp[df_temp["agg"] == agg]
                    if agg != "ens":
                        skills = df_temp_agg[df_temp_agg["metric"] == "crpss"][
                            "score"
                        ].tolist()

                        dict_skills = {
                            f"skill_{n_ens}": skill
                            for (n_ens, skill) in zip(n_ens_vec, skills)
                        }

                        # Add information to row
                        new_skill = {
                            "ens_method": ens_method,
                            "dataset": df_temp_agg["dataset"].iloc[0],
                            "nn": df_temp_agg["nn"].iloc[0],
                            "agg": df_temp_agg["agg"].iloc[0],
                            **dict_skills,
                            "avg_skill": np.mean(skills),
                        }

                        # Append to data frame
                        df_skills = pd.concat(
                            [
                                df_skills,
                                pd.DataFrame(new_skill, index=[0]),
                            ],
                            ignore_index=True,
                        )

                    scores = df_temp_agg[df_temp_agg["metric"] == "crps"][
                        "score"
                    ].tolist()

                    dict_scores = {
                        f"score_{n_ens}": score
                        for (n_ens, score) in zip(n_ens_vec, scores)
                    }

                    new_score = {
                        "ens_method": ens_method,
                        "dataset": df_temp_agg["dataset"].iloc[0],
                        "nn": df_temp_agg["nn"].iloc[0],
                        "agg": df_temp_agg["agg"].iloc[0],
                        **dict_scores,
                        "avg_score": np.mean(scores),
                    }

                    df_scores = pd.concat(
                        [
                            df_scores,
                            pd.DataFrame(new_score, index=[0]),
                        ],
                        ignore_index=True,
                    )

    return df_scores, df_skills


def get_final_scores_skills_table(
    paths: dict,
    drn_choice: dict,
    bqn_choice: dict | None,
    dataset_ls: list[str],
    ens_method_ls: list[str],
):
    score_vec = ["crps", "crpss", "me", "lgt", "cov", "a", "w"]
    distr_ls = ["drn", "bqn"]
    n_ens_vec = np.arange(start=2, stop=20 + 2, step=2)
    agg_meths = ["lp", "vi", "vi-a", "vi-w", "vi-aw"]

    # Get all skills table
    dataset_ls_norm = dataset_ls.copy()
    dataset_ls_norm.extend(["scen_1", "scen_4"])
    df_scores_norm, df_skills_norm = get_scores_skills_table(
        paths["norm"],
        dataset_ls_norm,
        score_vec,
        distr_ls,
        n_ens_vec,  # type: ignore
        agg_meths,
        ens_method_ls,
    )

    dataset_ls_tnorm = [
        "naval",
        "wine",
    ]
    df_scores_tnorm, df_skills_tnorm = get_scores_skills_table(
        paths["tnorm"],
        dataset_ls_tnorm,
        score_vec,
        distr_ls,
        n_ens_vec,  # type: ignore
        agg_meths,
        ens_method_ls,
    )

    df_scores_0tnorm, df_skills_0tnorm = get_scores_skills_table(
        paths["0tnorm"],
        dataset_ls,
        score_vec,
        distr_ls,
        n_ens_vec,  # type: ignore
        agg_meths,
        ens_method_ls,
    )

    df_scores_norm["distr"] = "norm"
    df_skills_norm["distr"] = "norm"
    df_scores_tnorm["distr"] = "tnorm"
    df_skills_tnorm["distr"] = "tnorm"
    df_scores_0tnorm["distr"] = "0tnorm"
    df_skills_0tnorm["distr"] = "0tnorm"

    df_scores = pd.concat(
        [df_scores_norm, df_scores_tnorm, df_scores_0tnorm], ignore_index=True
    )
    df_skills = pd.concat(
        [df_skills_norm, df_skills_tnorm, df_skills_0tnorm], ignore_index=True
    )

    if bqn_choice is None:
        bqn_choice = drn_choice

    drn_choice_tup = [
        ("drn", list(drn_choice.keys())[i], list(drn_choice.values())[i])
        for i in range(len(drn_choice))
    ]
    bqn_choice_tup = [
        ("bqn", list(bqn_choice.keys())[i], list(bqn_choice.values())[i])
        for i in range(len(bqn_choice))
    ]

    df_scores_filtered = pd.concat(
        [
            df_scores[
                df_scores[["nn", "dataset", "distr"]]
                .apply(tuple, axis=1)
                .isin(drn_choice_tup)
            ],
            df_scores[
                df_scores[["nn", "dataset", "distr"]]
                .apply(tuple, axis=1)
                .isin(bqn_choice_tup)
            ],
        ],
        ignore_index=True,
    )
    df_skills_filtered = pd.concat(
        [
            df_skills[
                df_skills[["nn", "dataset", "distr"]]
                .apply(tuple, axis=1)
                .isin(drn_choice_tup)
            ],
            df_skills[
                df_skills[["nn", "dataset", "distr"]]
                .apply(tuple, axis=1)
                .isin(bqn_choice_tup)
            ],
        ],
        ignore_index=True,
    )

    return df_scores_filtered, df_skills_filtered


def get_runtimes(
    data_path: str, dataset_ls: list[str], ens_method_ls: list[str]
) -> pd.DataFrame:
    """
    Read pickled DataFrames and concatenate them into a single DataFrame.
    Adds new columns with runtimes converted from nanoseconds to seconds.

    Parameters
    ----------
    data_path : str
        Path to directory containing the data.
    dataset_ls : list[str]
        List of strings representing names of the datasets.
    ens_method_ls : list[str]
        List of strings representing names of the ensemble methods.

    Returns
    -------
    df_final : pandas DataFrame
        A DataFrame containing columns with runtimes for training and
        prediction, the corresponding ensemble method, and the same
        runtimes in seconds.
    """

    df_final = pd.DataFrame()

    for dataset in dataset_ls:
        for ens_method in ens_method_ls:
            dir_path = os.path.join(data_path, dataset, ens_method)
            filename = f"runtime_{dataset}_{ens_method}.pkl"
            with open(os.path.join(dir_path, filename), "rb") as f:
                df_runtime = pickle.load(f)
            df_runtime["ens_method"] = ens_method

            df_final = pd.concat(
                [
                    df_final,
                    pd.DataFrame(df_runtime),
                ],
                ignore_index=True,
            )

    df_final["runtime_train_s"] = df_final["runtime_train"] / 1e9
    df_final["runtime_pred_s"] = df_final["runtime_pred"] / 1e9

    return df_final


def get_pi_coverage_table(
    paths: dict,
    drn_choice: dict,
    bqn_choice: dict,
    dataset_ls: list[str],
    score_vec: list[str],
    nn_vec: list[str],
    n_ens_vec: list[int],
    agg_meths: list[str],
    ens_method_ls: list[str],
) -> pd.DataFrame:
    """Returns a DataFrame containing the PI length and coverage scores.

    Parameters
    ----------
    data_path : string
        Path to eval_{dataset}_{ens_method}.pkl files
    dataset_ls : list[string]
        List containing the names of datasets to summarize
    score_vec : list[string]
        List containing the names of scores to summarize
        (e.g. crps, a, w, etc.)
    nn_vec : list[string]
        List containing the NN types to consider ("drn", "bqn")
    n_ens_vec : list[int]
        List containing ensemble sizes to consider
    agg_meths : list[string]
        List containing the aggregation methods to consider (e.g. "lp", "vi")
    ens_method_ls : list[string]
        List containing the names of ensembles to consider (e.g. "rand_init)

    Returns
    -------
    DataFrames
        Contains the skills for each ensemble size
    """
    df_results = pd.DataFrame()

    for ens_method in ens_method_ls:
        for dataset in dataset_ls:
            data_path = paths.get(drn_choice.get(dataset))
            df = get_panel_data(
                data_path,  # type: ignore
                [dataset],
                score_vec,
                nn_vec,
                n_ens_vec,
                agg_meths,
                ens_method,
            )

            filtered = df[df["metric"].isin(["lgt", "cov"])]
            filtered["ens_method"] = ens_method
            filtered["distr"] = drn_choice.get(dataset)

            df_results = pd.concat(
                [
                    df_results,
                    filtered,
                ],
                ignore_index=True,
            )

    return df_results


def get_pit(
    paths: dict,
    drn_choice: dict,
    data_ens_path: str,
    data_agg_path: str,
    ens_method_ls: list[str],
    dataset_ls: list[str],
    nn_vec: list[str],
    agg_meths: list[str],
):
    ### Simulation: PIT histograms ###
    # Network ensemble size
    n_ens = 2

    # Number of bins in histogram
    n_bins = 21

    df_plot = pd.DataFrame()

    # For-Loop over ens_methods
    for ens_method in ens_method_ls:
        # For-Loop over scenarios
        for dataset in dataset_ls:
            if dataset in ["protein", "year"]:
                temp_n_sim = 5
            else:
                temp_n_sim = 20
            ### Simulation: Load data ###
            data_path: str = paths.get(drn_choice.get(dataset))  # type: ignore
            filename = f"eval_{dataset}_{ens_method}.pkl"
            with open(os.path.join(data_path, filename), "rb") as f:
                df_scores = pickle.load(f)
            ### Initialization ###
            # Only scenario
            df_sc = df_scores[df_scores["model"] == dataset]

            if ens_method == "batchensemble":
                sample_sizes = {
                    "scen_1": (1000, 10_000),
                    "scen_4": (1000, 10_000),
                    "boston": (91, 51),
                    "concrete": (186, 103),
                    "energy": (139, 77),
                    "kin8nm": (1475, 819),
                    "naval": (2140, 1193),
                    "power": (1723, 957),
                    "protein": (8232, 4573),
                    "wine": (288, 160),
                    "yacht": (55, 31),
                }
                n_valid, n_test = sample_sizes.get(dataset)  # type: ignore
            else:
                n_valid = df_sc[df_sc["type"] != "ref"]["n_valid"].iloc[0]
                n_test = df_sc[df_sc["type"] != "ref"]["n_test"].iloc[0]

            # Index vector for validation and testing
            i_valid = list(range(n_valid))
            i_test = [max(i_valid) + el for el in list(range(n_test))]

            ### Get PIT values ###
            # List for PIT values
            pit_ls = {}

            # For-Loop over network variants
            for temp_nn in nn_vec:
                ### PIT values of ensemble member ###
                # Vector for PIT values
                temp_pit = []

                # For-Loop over ensemble member and simulation
                for i_rep in range(n_ens):
                    for i_sim in range(temp_n_sim):
                        # Load ensemble member
                        filename = f"{temp_nn}_sim_{i_sim}_ens_{i_rep}.pkl"  # noqa: E501
                        temp_data_ens_path = data_ens_path.replace(
                            "dataset", dataset
                        )
                        temp_data_ens_path_final = temp_data_ens_path.replace(
                            "ens_method", ens_method
                        )
                        with open(
                            os.path.join(temp_data_ens_path_final, filename),
                            "rb",
                        ) as f:
                            [pred_nn, _, _] = pickle.load(f)

                        # Read out
                        temp_pit.append(pred_nn["scores"]["pit"][i_test])

                # Save PIT
                pit_ls[f"{temp_nn}_ens"] = temp_pit

                ### Aggregation methods ###
                # For-Loop over aggregation methods
                for temp_agg in agg_meths:
                    # Vector for PIT values
                    temp_pit = []

                    # For-Loop over simulations
                    for i_sim in range(temp_n_sim):
                        # Load aggregated forecasts
                        filename = f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
                        temp_data_agg_path = data_agg_path.replace(
                            "dataset", dataset
                        )
                        temp_data_agg_path_final = temp_data_agg_path.replace(
                            "ens_method", ens_method
                        )
                        with open(
                            os.path.join(temp_data_agg_path_final, filename),
                            "rb",
                        ) as f:
                            pred_agg = pickle.load(f)

                        # Read out PIT-values
                        temp_pit.append(pred_agg["scores"]["pit"])

                    # Save PIT
                    pit_ls[f"{temp_nn}_{temp_agg}"] = temp_pit

            ### Calculate histograms ###
            # For-Loop over network variants and aggregation methods
            for temp_nn in nn_vec:
                for temp_agg in ["ens", *agg_meths]:
                    # Calculate histogram and read out values (see pit function)  # noqa: E501
                    temp_hist, temp_bin_edges = np.histogram(
                        pit_ls[f"{temp_nn}_{temp_agg}"],
                        bins=n_bins,
                        density=True,
                    )

                    new_row = {
                        "dataset": dataset,
                        "ens_method": ens_method,
                        "nn": temp_nn,
                        "agg": temp_agg,
                        "breaks": [temp_bin_edges],
                        "pit": [temp_hist],
                    }

                    df_plot = pd.concat(
                        [df_plot, pd.DataFrame(new_row, index=[0])],
                        ignore_index=True,
                    )

    return df_plot
