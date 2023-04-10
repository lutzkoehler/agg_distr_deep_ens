import json
import logging
import os
import pickle

import numpy as np

from ss_1_ensemble import train_valid_test_split

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    """
    Takes the original input of UCI datasets from
    https://github.com/yaringal/DropoutUncertaintyExps
    and saves the data in the pickle format used in this repo.
    """
    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    # Get all datasets to process
    dataset_ls = CONFIG["DATASET"]
    # Prepare path to data
    data_in_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["INPUT_DIR"],
        "dataset",
    )
    # For UCI Datasets, n_sim is default to 20
    n_sim = 20

    # Iterate over all datasets
    for dataset in dataset_ls:
        # Skip if dataset is a simulated scenario
        if dataset.startswith("scen"):
            continue

        # Use dataset directory
        temp_data_in_path = data_in_path.replace("dataset", dataset)

        # Iterate over all splits (named sim)
        for i_sim in range(n_sim):
            (
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
            ) = train_valid_test_split(temp_data_in_path, dataset, i_sim)

            X_train = np.vstack([X_train, X_valid])
            y_train = np.hstack([y_train, y_valid])

            ### Save data ###
            # Save ensemble member
            save_path = os.path.join(temp_data_in_path, f"sim_{i_sim}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump([X_train, y_train, X_test, y_test, None, None], f)
            log_message = f"Processed sim_{i_sim} of dataset {dataset.upper()}"
            logging.info(log_message)


if __name__ == "__main__":
    main()
