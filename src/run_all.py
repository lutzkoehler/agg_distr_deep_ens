import json
import os

import tensorflow as tf

import figures
import ss_1_ensemble
import ss_2_aggregation
import ss_3_scores

if __name__ == "__main__":
    #### Deactivate GPU usage ####
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "5"

    #### Limit cores to use ####
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(3)
    # tf.config.set_soft_device_placement(True)

    #### More specifications ####
    multi_run = True

    methods = [
        "rand_init",
        "bagging",
        "mc_dropout",
        "variational_dropout",
        "concrete_dropout",
        "bayesian",
        "batchensemble",
    ]
    # Run multiple ensemble methods
    if multi_run:
        for ens_method in methods:
            ### Get Config ###
            with open("src/config.json", "rt") as f:
                CONFIG = json.load(f)
            CONFIG["ENS_METHOD"] = ens_method
            with open("src/config.json", "wt") as f:
                json.dump(CONFIG, f)

            print(f"#### Running {ens_method} ####")
            # 1. Run ensemble prediction
            ss_1_ensemble.main()

            # 2. Run aggregation
            ss_2_aggregation.main()

            # 3. Score results
            ss_3_scores.main()

            # 4. Plot results
            figures.plot_panel_model()
            figures.plot_panel_boxplot()
            figures.plot_pit_ens()
            figures.plot_ensemble_members()
    # Run a single ensemble method
    else:
        # 1. Run ensemble prediction
        ss_1_ensemble.main()

        # 2. Run aggregation
        ss_2_aggregation.main()

        # 3. Score results
        ss_3_scores.main()

        # 4. Plot results
        figures.plot_panel_model()
        figures.plot_panel_boxplot()
        figures.plot_pit_ens()
        figures.plot_ensemble_members()
