## Function file
# Network variants for the simulation study

### Import ###
# Import basic functions
import multiprocessing

import os
import time
from typing import Any

import keras.backend as K
import numpy as np
from nptyping import NDArray, Float, Int
import scipy.stats as ss
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.backend import clear_session  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Concatenate, Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from fn_basic import update_hpar
from fn_eval import bern_quants, fn_scores_distr, fn_scores_ens

### DRN ###
## Distribution: Normal distribution
## Estimation: CRPS


def drn_pp(
    train: NDArray[Any, Float],
    test: NDArray[Any, Float],
    y_train: NDArray[Any, Float],
    y_test: NDArray[Any, Float],
    n_ens: int,
    i_valid: list[int] | None = None,
    nn_ls: dict[str, Any] | None = None,
    n_cores: float = np.nan,
    rpy_elements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Function for estimation and prediction

    Parameters
    ----------
    train/test : pd.DataFrame
        Training/Test data including predictors (n_train/test x n_preds
        data.frame)
    y_train/test
        Training/Test observations (n_train/test vector)
    n_ens
        Size of network ensemble
    i_valid
        Indices of validation data (n_valid vector), Default: NULL -> Use
        fraction of training data
    nn_ls
        List that may contain the following variables:
        lr_adam
            Learning rate in Adam-Optimizer (scalar), Default: 5e-4
        n_epochs
            Number of epochs (integer), Default: 150
        n_patience
            Patience in callbacks (integer), Default: 10
        n_batch
            Size of batches is equal to n_train/n_batch (input parameter
            batch_size), Default: 64
        lay1
            Number of nodes in first and second (use half) hidden layer
            (integer), Default: 64 -> 32 in second
        actv
            Activation function of non-output layers (string), Default:
            "softplus"
        nn_verbose
            Query, if network output should be shown (logical), Default: 0
    n_cores
        Number of cores used in keras (integer), Default: NULL -> Use one less
        than available

    Returns
    -------
    res
        List containing:
        f
            Distributional forecasts (i.e. parameters) (n_test x 2 matrix)
        nn_ls
            Hyperparameters (list)
        n_train
            Number of training samples (integer)
        n_valid
            Number of validation samples (integer)
        n_test
            Number of test samples (integer)
        runtime_est
            Estimation time (numeric)
        runtime_pred
            Prediction time (numeric)
        scores
            Data frame containing (n x 6 data frame):
            pit
                PIT values of DRN forecasts (n vector)
            crps
                CRPS of DRN forecasts (n vector)
            logs
                Log-Score of DRN forecasts (n vector)
            lgt
                Length of DRN prediction interval (n vector)
            e_md
                Bias of median forecast (n vector)
            e_me
                Bias of mean forecast (n vector)
    """

    if i_valid is None:
        i_valid = []
    if nn_ls is None:
        nn_ls = {}

    ### Initiation ###
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.keras.backend.set_floatx("float64")

    # Number of cores
    if n_cores is None:
        n_cores = multiprocessing.cpu_count() / 2 - 1
    # Set number of cores
    n_cores_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=int(n_cores),
        inter_op_parallelism_threads=int(n_cores),
    )
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=n_cores_config)
    )

    ### Hyperparameter ###
    # Hyperparameters and their default values
    hpar_ls: dict[str, Any] = {
        "lr_adam": 5e-4,  # -1 for Adam-default
        "n_epochs": 150,
        "n_patience": 10,
        "n_batch": 64,
        "lay1": 64,
        "actv": "softplus",
        "nn_verbose": 0,
    }

    # Update hyperparameters
    nn_ls = update_hpar(hpar_ls=hpar_ls, in_ls=nn_ls)

    # Custom optizimer
    if nn_ls["lr_adam"] == -1:
        custom_opt = "adam"
    else:
        custom_opt = Adam(learning_rate=nn_ls["lr_adam"])

    # Get standard logistic distribution
    tfd = tfp.distributions.Normal(
        loc=np.float64(0), scale=np.float64(1)
    )  # , dtype=tf.float64)

    # Custom loss function
    def custom_loss(y_true, y_pred):
        # Get location and scale
        mu = K.dot(
            y_pred, K.constant(np.r_[1, 0], shape=(2, 1), dtype=np.float64)
        )
        sigma = K.dot(
            y_pred, K.constant(np.r_[0, 1], shape=(2, 1), dtype=np.float64)
        )

        # Standardization
        z_y = (y_true - mu) / sigma

        # Calculate CRPS
        res = sigma * (
            z_y * (2 * tfd.cdf(z_y) - 1)
            + 2 * tfd.prob(z_y)
            - 1 / np.sqrt(np.pi)
        )

        # Calculate mean
        res = K.mean(res)

        # Return mean CRPS
        return res

    ### Data preparation ###
    # Divide data in training and validation set
    i_train: NDArray[Any, Int] = np.delete(
        arr=np.arange(stop=train.shape[0], dtype=np.int64),  # type: ignore
        obj=i_valid,
    )

    # Read out set sizes
    n_train = len(i_train)
    n_valid = len(i_valid)
    n_test = test.shape[0]

    # Save center and scale parameters
    tr_center: NDArray[Any, Float] = np.mean(train[i_train, :], axis=0)
    # tr_scale: NDArray[Any, Float] = np.nan_to_num(
    #     x=np.std(train[i_train, :], axis=0), nan=1
    # )
    tr_scale: NDArray[Any, Float] = np.std(train[i_train, :], axis=0)

    # Scale training data
    X_train = (train[i_train, :] - tr_center) / tr_scale

    # Scale validation data with training data attributes
    X_valid = (train[np.array(i_valid), :] - tr_center) / tr_scale

    # Scale data for prediction
    X_pred = (test - tr_center) / tr_scale

    ### Build network ###
    # Input
    input = Input(shape=X_train.shape[1], name="input", dtype="float64")

    # Hidden layers
    hidden_1 = Dense(
        units=nn_ls["lay1"], activation=nn_ls["actv"], dtype="float64"
    )(input)
    hidden_2 = Dense(
        units=nn_ls["lay1"] / 2, activation=nn_ls["actv"], dtype="float64"
    )(hidden_1)

    # Different activation functions for output
    loc_out = Dense(units=1, dtype="float64")(hidden_2)
    scale_out = Dense(units=1, activation="softplus", dtype="float64")(
        hidden_2
    )

    # Concatenate output
    output = Concatenate()([loc_out, scale_out])

    # Define model
    model = Model(inputs=input, outputs=output)

    ### Estimation ###
    # Compile model
    model.compile(optimizer=custom_opt, loss=custom_loss)

    # Take time
    start_tm = time.time_ns()

    # Fit model
    model.fit(
        x=X_train,
        y=y_train[i_train],
        epochs=nn_ls["n_epochs"],
        batch_size=nn_ls["n_batch"],
        validation_data=(X_valid, y_train[i_valid]),
        verbose=nn_ls["nn_verbose"],
        callbacks=EarlyStopping(
            patience=nn_ls["n_patience"],
            restore_best_weights=True,
            monitor="val_loss",
        ),
    )

    # Take time
    end_tm = time.time_ns()

    # Time needed
    runtime_est = end_tm - start_tm

    ### Prediciton ###
    # Take time
    start_tm = time.time_ns()

    # Predict parameters of distributional forecasts (on scaled data)
    f: NDArray[Any, Float] = model.predict(X_pred)

    # Take time
    end_tm = time.time_ns()

    # Time needed
    runtime_pred = end_tm - start_tm

    # Delete model
    del input, hidden_1, hidden_2, loc_out, scale_out, output, model

    # Clear memory and session
    clear_session()

    # Delete data
    del X_train, X_valid, X_pred

    ### Evaluation ###
    # Calculate evaluation measres of DRN forecasts
    scores = fn_scores_distr(
        f=f, y=y_test, distr="norm", rpy_elements=rpy_elements
    )

    ### Output ###
    # Output
    return {
        "f": f,
        "nn_ls": nn_ls,
        "scores": scores,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "runtime_est": runtime_est,
        "runtime_pred": runtime_pred,
    }


### BQN ###
# Distribution: Quantile function of Bernstein polynomials
# Estimation: Quantile loss


def bqn_pp(
    train: NDArray[Any, Float],
    test: NDArray[Any, Float],
    y_train: NDArray[Any, Float],
    y_test: NDArray[Any, Float],
    n_ens: int,
    i_valid: list[int] | None = [],
    q_levels: NDArray[Any, Float] | None = None,
    nn_ls: dict[str, Any] | None = {},
    n_cores: float = np.nan,
    rpy_elements: dict[str, Any] | None = None,
):
    """
    Function including estimation and prediction

    Parameters
    ----------
    train/test : pd.DataFrame
        Training/Test data including predictors (n_train/test x n_preds
        data.frame)
    y_train/test
        Training/Test observations (n_train/test vector)
    i_valid
        Indices of validation data (n_valid vector), Default: NULL -> Use
        fraction of training data
    q_levels
        Quantile levels used for output and evaluation (n_q probability
        vector), Default: NULL -> 99 member, incl. median
    n_ens
        Size of network ensemble
    nn_ls
        List that may contain the following variables:
        p_degree
            Degree of Bernstein polynomials (integer), Default: 12
        n_q
            Number of equidistant quantile levels used in loss function
            (integer), Default: 99 (steps of 1%)
        lr_adam
            Learning rate in Adam-Optimizer (scalar), Default: 5e-4
        n_epochs
            Number of epochs (integer), Default: 150
        n_patience
            Patience for early stopping (integer), Default: 10
        n_batch
            Size of batches is equal to n_train/n_batch (input parameter
            batch_size), Default: 64
        lay1
            Number of nodes in first hidden layer (integer), Default: 48
        actv
            Activation function of non-output layers (string)
            Default: "softplus"
        actv_out
            Activation function of output layer exlcuding alpha_0 (string),
            Default: "softplus"
        nn_verbose
            Query, if network output should be shown (logical), Default: 0
    n_cores
        Number of cores used in keras (integer), Default: NULL -> Use one less
        than available

    Returns
    -------
    res
        List containing:
        f
            BQN forecasts (i.e. quantiles) based on q_levels (n x n_q matrix)
        alpha
            BQN coefficients (n x p_degree matrix)
        nn_ls
            Hyperparameters (list)
        n_train
            Number of training samples (integer)
        n_valid
            Number of validation samples (integer)
        n_test
            Number of test samples (integer)
        runtime_est
            Estimation time (numeric)
        runtime_pred
            Prediction time (numeric)
        scores
            Data frame containing (n x 6 data frame):
            rank
                Ranks of observations in BQN forecasts (n vector)
            crps
                CRPS of BQN forecasts (n vector)
            logs
                Log-Score of BQN forecasts (n vector)
            lgt
                Length of BQN prediction interval (n vector)
            e_md
                Bias of median forecast (n vector)
            e_me
                Bias of mean forecast (n vector)
    """

    if i_valid is None:
        i_valid = []
    if nn_ls is None:
        nn_ls = {}

    ### Initiation ###
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.keras.backend.set_floatx("float64")

    # Number of cores
    if n_cores is None:
        n_cores = multiprocessing.cpu_count() / 2 - 1
    # Set number of cores
    n_cores_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=int(n_cores),
        inter_op_parallelism_threads=int(n_cores),
    )
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=n_cores_config)
    )

    # If not given use equidistant quantiles (multiple of ensemble coverage,
    # incl. median)
    if q_levels is None:
        step = 1 / 100
        q_levels = np.arange(start=1 / 100, stop=1 + step, step=step)

    ### Hyperparameter ###
    # Hyperparameters and their default values
    hpar_ls: dict[str, Any] = {
        "p_degree": 12,
        "n_q": 99,
        "lr_adam": 5e-4,  # -1 for Adam-default
        "n_epochs": 150,
        "n_patience": 10,
        "n_batch": 64,
        "lay1": 48,
        "actv": "softplus",
        "actv_out": "softplus",
        "nn_verbose": 0,
    }

    # Update hyperparameters
    nn_ls = update_hpar(hpar_ls=hpar_ls, in_ls=nn_ls)

    # Calculate equidistant quantile levels for loss function
    q_levels_loss = np.arange(
        start=1 / (nn_ls["n_q"] + 1),
        stop=(nn_ls["n_q"] + 1) / (nn_ls["n_q"] + 1),
        step=1 / (nn_ls["n_q"] + 1),
    )

    # Basis of Bernstein polynomials evaluated at quantile levels
    B = np.apply_along_axis(
        func1d=ss.binom.pmf,
        arr=np.reshape(
            np.arange(start=0, stop=nn_ls["p_degree"] + 1, step=1),
            newshape=(1, nn_ls["p_degree"] + 1),
        ),
        axis=0,
        p=q_levels_loss,
        n=nn_ls["p_degree"],
    )

    # Quantile loss functions (for neural network)
    def qt_loss(y_true, y_pred):
        """Quantile loss for BQN network

        Quantile Loss: L(yp_i, y_i) = max[ q(y_i - yp_i), (q - 1)(y_i - yp_i) ]

        Parameters
        ----------
        y_true : array(n x 1)
            True observation to predict
        y_pred : array(n x n_coeff)
            Predicted Bernstein coefficients and increments

        Returns
        -------
        scalar
            Mean quantile loss across all observations
        """
        # Quantiles calculated via basis and increments -> q_shape: (64,99)
        q = K.dot(K.cumsum(y_pred, axis=1), K.constant(B.T, dtype=np.float64))

        # Calculate individual quantile scores
        err = y_true - q  # err_shape (64,99)
        e1 = err * K.constant(
            value=q_levels_loss, shape=(1, nn_ls["n_q"]), dtype=np.float64
        )  # e1_shape (64,99)
        e2 = err * K.constant(
            value=q_levels_loss - 1, shape=(1, nn_ls["n_q"]), dtype=np.float64
        )

        # Find correct values (max) and return mean
        return K.mean(
            K.maximum(e1, e2), axis=1
        )  # max_shape (64,99) - mean_shape (64,)

    # Custom optizimer
    if nn_ls["lr_adam"] == -1:
        custom_opt = "adam"
    else:
        custom_opt = Adam(learning_rate=nn_ls["lr_adam"])

    ### Data preparation ###
    # Divide data in training and validation set
    i_train = np.delete(
        arr=np.arange(train.shape[0]),
        obj=i_valid,
        axis=0,
    )

    # Read out set sizes
    n_train = len(i_train)
    n_valid = len(i_valid)
    n_test = test.shape[0]

    # Save center and scale parameters
    tr_center = np.mean(train[i_train, :], axis=0)
    tr_scale = np.std(train[i_train, :], axis=0)

    # Scale training data
    X_train = (train[i_train, :] - tr_center) / tr_scale

    # Scale validation data with training data attributes
    X_valid = (train[np.array(i_valid), :] - tr_center) / tr_scale

    # Scale data for prediction
    X_pred = (test - tr_center) / tr_scale

    ### Build network ###
    # Input
    input = Input(shape=X_train.shape[1], name="input", dtype="float64")

    # Hidden layers
    hidden_1 = Dense(
        units=nn_ls["lay1"], activation=nn_ls["actv"], dtype="float64"
    )(input)
    hidden_2 = Dense(
        units=nn_ls["lay1"] / 2, activation=nn_ls["actv"], dtype="float64"
    )(hidden_1)

    # Different activation functions for output (alpha_0 and positive
    # increments)
    alpha0_out = Dense(units=1, dtype="float64")(hidden_2)
    alphai_out = Dense(
        units=nn_ls["p_degree"], activation="softplus", dtype="float64"
    )(hidden_2)

    # Concatenate output
    output = Concatenate()([alpha0_out, alphai_out])

    # Define model
    model = Model(inputs=input, outputs=output)

    ### Estimation ###
    # Compile model
    # run_eagerly to help debug (default: False)
    model.compile(optimizer=custom_opt, loss=qt_loss, run_eagerly=True)

    # Take time
    start_tm = time.time_ns()

    # Fit model
    model.fit(
        x=X_train,
        y=y_train[i_train],
        epochs=nn_ls["n_epochs"],
        batch_size=nn_ls["n_batch"],
        validation_data=(X_valid, y_train[i_valid]),
        verbose=nn_ls["nn_verbose"],
        callbacks=EarlyStopping(
            patience=nn_ls["n_patience"],
            restore_best_weights=True,
            monitor="val_loss",
        ),
    )

    # Take time
    end_tm = time.time_ns()

    # Time needed
    runtime_est = end_tm - start_tm

    ### Prediciton ###
    # Take time
    start_tm = time.time_ns()

    # Predict coefficients of Bernstein polynomials
    coeff_bern = model.predict(X_pred)

    # Take time
    end_tm = time.time_ns()

    # Time needed
    runtime_pred = end_tm - start_tm

    # Delete model
    del input, hidden_1, hidden_2, alpha0_out, alphai_out, output, model

    # Clear memory and session
    clear_session()

    # Delete data
    del X_train, X_valid, X_pred

    # Accumulate increments
    coeff_bern = np.cumsum(coeff_bern, axis=1)

    ### Evaluation ###
    # Sum up calcuated quantiles (Sum of basis at quantiles times coefficients)
    q = bern_quants(alpha=coeff_bern, q_levels=q_levels)
    # Calculate evaluation measres of DRN forecasts
    scores = fn_scores_ens(
        ens=q,
        y=y_test,
        skip_evals=["e_me"],
        scores_ens=True,
        rpy_elements=rpy_elements,
    )

    # Transform ranks to n_(ens+1) bins (for multiples of (n_ens+1) exact)
    if q.shape[1] != n_ens:
        scores["rank"] = np.ceil(
            scores["rank"] * (n_ens + 1) / (q.shape[1] + 1)
        )

    # Calculate bias of mean forecast (formula given)
    scores["e_me"] = np.mean(coeff_bern, axis=1) - y_test

    ### Output ###
    # Output
    return {
        "f": q,
        "alpha": coeff_bern,
        "nn_ls": nn_ls,
        "scores": scores,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "runtime_est": runtime_est,
        "runtime_pred": runtime_pred,
    }
