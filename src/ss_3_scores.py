## Simulation study: Script 3
# Scores of deep ensembles and aggregated forecasts

import json
import logging
import os
import pickle
from time import time_ns

import numpy as np
import pandas as pd

from fn_eval import fn_cover


def main():
    #### Deactivate GPU usage ####
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "5"

    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    # Take time
    start_time = time_ns()

    ### Settings ###
    ens_method = CONFIG["ENS_METHOD"]
    # Path of simulated data
    data_raw_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["INPUT_DIR"],
        "dataset",
    )

    # Path of deep ensemble forecasts
    data_ens_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["ENSEMBLE_F"],
    )

    # Path of aggregated forecasts
    data_agg_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["AGG_F"],
    )

    # Path of results
    data_out_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
    )

    ### Initialize ###
    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulations
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # (Maximum) number of network ensembles
    n_rep = CONFIG["PARAMS"]["N_ENS"]

    # Ensemble sizes to be combined
    if n_rep > 1:
        step_size = 2
        n_ens_vec = np.arange(
            start=step_size, stop=n_rep + step_size, step=step_size
        )
    else:
        n_ens_vec = [0]

    # Network types
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # Aggregation methods
    agg_meths_ls = CONFIG["PARAMS"]["AGG_METHS_LS"]

    # To evaluate
    sr_eval = ["crps", "logs", "lgt", "cov", "mae", "me", "rmse"]

    # Skill scores
    sr_skill = ["crps", "logs", "mae", "rmse"]
    # sr_skill = ["crps"]

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
    df_runtime = pd.DataFrame(
        columns=[
            "i_sim",
            "i_ens",
            "nn",
            "dataset",
            "runtime_train",
            "runtime_pred",
        ]
    )

    # For-Loop over scenarios and simulations
    for dataset in dataset_ls:
        # Check if scenario or UCI dataset
        if dataset.startswith("scen"):
            temp_n_sim = n_sim
            optimal_score_available = True
        elif dataset in ["protein", "year"]:
            temp_n_sim = 5
            optimal_score_available = False
        else:
            temp_n_sim = 20
            optimal_score_available = False

        # For-Loop over network types
        for temp_nn in nn_vec:
            agg_meths = agg_meths_ls[temp_nn]

            for i_sim in range(temp_n_sim):
                ### Calculate optimal scores based on data generation ###
                if (temp_nn == nn_vec[0]) and optimal_score_available:
                    # Load data
                    filename = f"sim_{i_sim}.pkl"
                    temp_data_raw_path = data_raw_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_raw_path, filename), "rb"
                    ) as f:
                        (
                            _,  # X_train
                            _,  # y_train
                            _,  # X_test
                            _,  # y_test
                            _,  # f_opt
                            scores_opt,
                        ) = pickle.load(f)

                    # Make entry for optimal forecast
                    new_row = {
                        "model": dataset,
                        "n_sim": i_sim,
                        "nn": "ref",
                        "type": "ref",
                    }

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

                ### Read out scores with ensemble size 1 ###
                # For-Loop over repetitions
                temp_dict = {}
                for i_rep in range(n_rep):
                    # Write in data frame
                    new_row = {
                        "model": dataset,
                        "n_sim": i_sim,
                        "nn": temp_nn,
                        "type": "ind",
                        "n_ens": 1,
                        "n_rep": i_rep,
                    }

                    # Load ensemble member
                    filename = os.path.join(
                        f"{temp_nn}_sim_{i_sim}_ens_{i_rep}.pkl",  # noqa: E501
                    )
                    temp_data_ens_path = data_ens_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_ens_path, filename), "rb"
                    ) as f:
                        pred_nn, _, _ = pickle.load(
                            f
                        )  # pred_nn, y_valid, y_test

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
                        new_row[temp] = temp_dict[temp]

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

                    ### Save runtime ###
                    runtime_train = pred_nn["runtime_est"]
                    runtime_pred = pred_nn["runtime_pred"]

                    new_row = {
                        "i_sim": i_sim,
                        "i_ens": i_rep,
                        "nn": temp_nn,
                        "dataset": dataset,
                        "runtime_train": runtime_train,
                        "runtime_pred": runtime_pred,
                    }

                    # Append to data frame
                    df_runtime = pd.concat(
                        [
                            df_runtime,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

                ### Read out scores for aggregated ensembles ###
                # For-Loop over aggregation methods
                for temp_agg in agg_meths:
                    # For-Loop over number of aggregated members
                    for i_ens in n_ens_vec:
                        # Write in data frame
                        new_row = {
                            "model": dataset,
                            "n_sim": i_sim,
                            "nn": temp_nn,
                            "type": temp_agg,
                            "n_ens": i_ens,
                            "n_rep": 0,
                        }

                        # Get set sizes
                        for temp in ["n_train", "n_valid", "n_test"]:
                            new_row[temp] = temp_dict[temp]

                        # Load aggregated forecasts
                        filename = os.path.join(
                            f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{i_ens}.pkl",  # noqa: E501
                        )
                        temp_data_agg_path = data_agg_path.replace(
                            "dataset", dataset
                        )
                        with open(
                            os.path.join(temp_data_agg_path, filename), "rb"
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

                        # Calculate skill scores if optimal score is available
                        if optimal_score_available:
                            # For-Loop over skill scores
                            for temp_sr in sr_skill:
                                # Reference is given by mean score of network
                                # ensemble members
                                s_ref = df_scores[
                                    (df_scores["model"] == dataset)
                                    & (df_scores["n_sim"] == i_sim)
                                    & (df_scores["nn"] == temp_nn)
                                    & (df_scores["type"] == "ind")
                                    & (df_scores["n_rep"] <= i_ens)
                                ][temp_sr].mean()

                                # Score of optimal forecast
                                s_opt = df_scores[
                                    (df_scores["model"] == dataset)
                                    & (df_scores["n_sim"] == i_sim)
                                    & (df_scores["nn"] == "ref")
                                ][temp_sr]

                                if s_ref == np.Inf:
                                    log_message = (
                                        f"{ens_method.upper()}, "
                                        f"{dataset.upper()}, "
                                        f"{temp_nn.upper()}: "
                                        "Ensemble member with infinite "
                                        f"score ({temp_sr})"
                                    )
                                    logging.warning(log_message)
                                # Calculate skill
                                temp_entry = (s_ref - new_row[temp_sr]) / (
                                    s_ref - s_opt
                                )
                                new_row[f"{temp_sr}s"] = temp_entry.iloc[0]

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
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_out_path = data_out_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_out_path, filename), "wb") as f:
            pickle.dump(df_scores, f)
        # Save runtime dataframe
        filename = f"runtime_{dataset}_{ens_method}.pkl"
        temp_data_out_path = data_out_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_out_path, filename), "wb") as f:
            pickle.dump(df_runtime, f)

        log_message = (
            f"{dataset.upper()}, {ens_method}: Finished scoring of {filename} "
            f"- {(end_time - start_time) / 1e+9:.2f}s"
        )
        logging.info(log_message)


if __name__ == "__main__":
    ### Set log Level ###
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # np.seterr(all="raise")
    main()
