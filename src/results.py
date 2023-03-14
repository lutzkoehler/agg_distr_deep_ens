import os
import pickle

import pandas as pd


def get_panel_data(
    data_path, dataset_ls, score_vec, nn_vec, n_ens_vec, agg_meths, ens_method
):
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


def get_best_result_table(
    data_path, nn_vec, ens_method_ls, agg_meths, dataset_ls, n_ens_vec
):
    score_vec = ["crps", "crpss", "me", "lgt", "cov", "a", "w"]

    results = {"drn_results": None, "bqn_results": None}

    for ens_method in ens_method_ls:
        temp_data_path = os.path.join(data_path, ens_method)
        df = get_panel_data(
            temp_data_path,
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
