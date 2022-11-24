## Simulation study: Script 3
# Scores of deep ensembles and aggregated forecasts

import os
import pickle
from time import time_ns

import numpy as np
import pandas as pd

from fn_eval import fn_cover


def main():
    # Take time
    start_time = time_ns()

    ### Settings ###
    # Path of simulated data
    data_sim_path = os.path.join("data", "ss_data")

    # Path of deep ensemble forecasts
    data_ens_path = os.path.join("data", "model")

    # Path of aggregated forecasts
    data_agg_path = os.path.join("data", "agg")

    # Path of results
    data_out_path = "data"

    ### Initialize ###
    # Models considered
    # scenario_vec = range(1, 7, 1)
    scenario_vec = [1, 4]

    # Number of simulations
    # n_sim = 50
    n_sim = 10

    # Ensemble sizes to be combined
    n_ens_vec = np.arange(start=2, stop=12, step=2)

    # (Maximum) number of network ensembles
    n_rep = np.max(n_ens_vec)

    # Network types
    nn_vec = ["drn", "bqn"]  # For now without "hen"

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

    # To evaluate
    sr_eval = ["crps", "logs", "lgt", "cov", "mae", "me", "rmse"]

    # Skill scores
    sr_skill = ["crps", "logs", "mae", "rmse"]

    # Vector of column names
    col_vec_pp = (
        [
            "model",
            "n_sim",
            "n_test",
            "n_train",
            "n_valid",
            "nn",
            "type",
            "n_ens",
            "n_rep",
            "a",
            "w",
        ]
        + sr_eval
        + [f"{entry}s" for entry in sr_skill]
    )

    ### Create data frame ###
    df_scores = pd.DataFrame(columns=col_vec_pp)

    # For-Loop over network types
    for temp_nn in nn_vec:
        agg_meths = agg_meths_ls[temp_nn]

        # For-Loop over scenarios and simulations
        for i_scenario in scenario_vec:
            for i_sim in range(n_sim):
                if temp_nn == nn_vec[0]:
                    # Load data
                    filename = f"scen_{i_scenario}_sim_{i_sim}.pkl"
                    with open(
                        os.path.join(data_sim_path, filename), "rb"
                    ) as f:
                        (
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            f_opt,
                            scores_opt,
                        ) = pickle.load(f)

                    # Make entry for optimal forecast
                    new_row = {}
                    new_row["model"] = i_scenario
                    new_row["n_sim"] = i_sim
                    new_row["nn"] = "ref"
                    new_row["type"] = "ref"

                    # For-Loop over evaluation measures
                    for temp_sr in sr_eval:
                        # Depending on measure
                        if temp_sr == "mae":
                            new_row[temp_sr] = np.mean(
                                np.abs(scores_opt["e_md"])
                            )
                        elif temp_sr == "me":
                            new_row[temp_sr] = np.mean(scores_opt["e_md"])
                        elif temp_sr == "rmse":
                            new_row[temp_sr] = np.sqrt(
                                np.mean(scores_opt["e_me"] ** 2)
                            )
                        elif temp_sr == "cov":
                            new_row[temp_sr] = np.mean(
                                fn_cover(scores_opt["pit"])
                            )
                        else:
                            new_row[temp_sr] = np.mean(scores_opt[temp_sr])

                    # Append to data frame
                    df_scores = pd.concat(
                        [
                            df_scores,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

                # For-Loop over repetitions
                temp_dict = {}
                for i_rep in range(n_rep):
                    # Write in data frame
                    new_row = {}
                    new_row["model"] = i_scenario
                    new_row["n_sim"] = i_sim
                    new_row["nn"] = temp_nn
                    new_row["type"] = "ind"
                    new_row["n_ens"] = 1
                    new_row["n_rep"] = i_rep

                    # Load ensemble member
                    filename = os.path.join(
                        f"{temp_nn}_scen_{i_scenario}_sim_{i_sim}_ens_{i_rep}.pkl",  # noqa: E501
                    )
                    with open(
                        os.path.join(data_ens_path, filename), "rb"
                    ) as f:
                        pred_nn, y_valid, y_test = pickle.load(f)

                    # Get set size of first repetition
                    for temp in ["n_train", "n_valid", "n_test"]:
                        # Save set sizes
                        if i_rep == 0:
                            # Read out set sizes
                            temp_dict[temp] = pred_nn[temp]

                            # Calculate actual test set size
                            if temp == "n_test":
                                temp_dict[temp] = (
                                    pred_nn["n_test"] - pred_nn["n_valid"]
                                )

                        # Read out set sizes
                        new_row[temp] = temp_dict[temp]  # ignore

                    # Cut validation data from scores
                    pred_nn["scores"] = pred_nn["scores"].drop(
                        range(pred_nn["n_valid"])
                    )

                    # For-Loop over evaluation measures
                    for temp_sr in sr_eval:
                        # Depending on measure
                        if temp_sr == "mae":
                            new_row[temp_sr] = np.mean(
                                np.abs(pred_nn["scores"]["e_md"])
                            )
                        elif temp_sr == "me":
                            new_row[temp_sr] = np.mean(
                                pred_nn["scores"]["e_md"]
                            )
                        elif temp_sr == "rmse":
                            new_row[temp_sr] = np.sqrt(
                                np.mean(pred_nn["scores"]["e_me"] ** 2)
                            )
                        elif temp_sr == "cov":
                            new_row[temp_sr] = np.mean(
                                fn_cover(pred_nn["scores"]["pit"])
                            )
                        else:
                            new_row[temp_sr] = np.mean(
                                pred_nn["scores"][temp_sr]
                            )

                    # Append to data frame
                    df_scores = pd.concat(
                        [
                            df_scores,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

                # For-Loop over aggregation methods
                for temp_agg in agg_meths:
                    # For-Loop over number of aggregated members
                    for i_ens in n_ens_vec:
                        # Write in data frame
                        new_row = {}
                        new_row["model"] = i_scenario
                        new_row["n_sim"] = i_sim
                        new_row["nn"] = temp_nn
                        new_row["type"] = temp_agg
                        new_row["n_ens"] = i_ens
                        new_row["n_rep"] = 0

                        # Get set sizes
                        for temp in ["n_train", "n_valid", "n_test"]:
                            new_row[temp] = temp_dict[temp]

                        # Load aggregated forecasts
                        filename = os.path.join(
                            f"{temp_nn}_scen_{i_scenario}_sim_{i_sim}_{temp_agg}_ens_{i_ens}.pkl",  # noqa: E501
                        )
                        with open(
                            os.path.join(data_agg_path, filename), "rb"
                        ) as f:
                            pred_agg = pickle.load(f)

                        # Estimated weights and intercept
                        if temp_agg == "vi-w":
                            new_row["w"] = pred_agg["w"]
                        elif temp_agg == "vi-a":
                            new_row["a"] = pred_agg["a"]
                        elif temp_agg == "vi-aw":
                            new_row["a"] = pred_agg["a"]
                            new_row["w"] = pred_agg["w"]

                        # For-Loop over evaluation measures
                        for temp_sr in sr_eval:
                            # Depending on measure
                            if temp_sr == "mae":
                                new_row[temp_sr] = np.mean(
                                    np.abs(pred_agg["scores"]["e_md"])
                                )
                            elif temp_sr == "me":
                                new_row[temp_sr] = np.mean(
                                    pred_agg["scores"]["e_md"]
                                )
                            elif temp_sr == "rmse":
                                new_row[temp_sr] = np.sqrt(
                                    np.mean(pred_agg["scores"]["e_me"] ** 2)
                                )
                            elif temp_sr == "cov":
                                new_row[temp_sr] = np.mean(
                                    fn_cover(pred_agg["scores"]["pit"])
                                )
                            else:
                                new_row[temp_sr] = np.mean(
                                    pred_agg["scores"][temp_sr]
                                )

                        # For-Loop over skill scores
                        for temp_sr in sr_skill:
                            # Reference is given by mean score of network
                            # ensemble members
                            s_ref = df_scores[
                                (df_scores["model"] == i_scenario)
                                & (df_scores["n_sim"] == i_sim)
                                & (df_scores["nn"] == temp_nn)
                                & (df_scores["type"] == "ind")
                                & (df_scores["n_rep"] <= i_ens)
                            ][temp_sr].mean()

                            # Score of optimal forecast
                            s_opt = df_scores[
                                (df_scores["model"] == i_scenario)
                                & (df_scores["n_sim"] == i_sim)
                                & (df_scores["nn"] == "ref")
                            ][temp_sr]

                            # Calculate skill
                            new_row[f"{temp_sr}s"] = (
                                s_ref - new_row[temp_sr]
                            ) / (s_ref - s_opt)

                        # Append to data frame
                        df_scores = pd.concat(
                            [
                                df_scores,
                                pd.DataFrame(new_row, index=[0]),
                            ],
                            ignore_index=True,
                        )
    # Take time
    end_time = time_ns()

    ### Save ###
    filename = "eval_ss.pkl"
    with open(os.path.join(data_out_path, filename), "wb") as f:
        pickle.dump(df_scores, f)
    print(f"Finished scoring - Results stored in file {filename}")
    print(f"Total time spent: {(end_time - start_time) / 1e+9}s")


if __name__ == "__main__":
    # np.seterr(all="raise")
    main()
