# Build different model architectures

from typing import Any
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.layers import Concatenate, Dense, Input  # type: ignore


def get_drn_base_model(
    input_length: int, hpar_ls: dict[str, Any], dtype: str = "float32"
) -> Model:
    """Construct and return DRN base model

    Architecture:

    Layer (type)                Output Shape    Param #  Connected to
    =========================================================================
    input (InputLayer)          [(None, 5)]     0        []
    dense_1 (Dense)             (None, 64)      384      ['input[0][0]']
    dense (Dense)               (None, 32)      2080     ['dense_1[0][0]']
    dense_2 (Dense)             (None, 1)       33       ['dense[0][0]']
    dense_3 (Dense)             (None, 1)       33       ['dense[0][0]']
    concatenate (Concatenate)   (None, 2)       0        ['dense_2[0][0]',
                                                            'dense_3[0][0]']

    Parameters
    ----------
    input_length : int
        Number of predictors
    hpar_ls : dict[str, Any]
        Contains several hyperparameter
    dtype : str
        Should be either float32 or float64

    Returns
    -------
    Model
        _description_
    """
    tf.keras.backend.set_floatx(dtype)

    ### Build network ###
    # Input
    input = Input(shape=input_length, name="input", dtype=dtype)

    # Hidden layers
    hidden_1 = Dense(
        units=hpar_ls["lay1"], activation=hpar_ls["actv"], dtype=dtype
    )(input)
    hidden_2 = Dense(
        units=hpar_ls["lay1"] / 2, activation=hpar_ls["actv"], dtype=dtype
    )(hidden_1)

    # Different activation functions for output
    loc_out = Dense(units=1, dtype=dtype)(hidden_2)
    scale_out = Dense(units=1, activation="softplus", dtype=dtype)(hidden_2)

    # Concatenate output
    output = Concatenate()([loc_out, scale_out])

    # Define model
    model = Model(inputs=input, outputs=output)

    # Return model
    return model


def get_bqn_base_model(
    input_length: int, hpar_ls: dict[str, Any], dtype: str = "float32"
) -> Model:
    """Construct and return BQN base model

    Architecture:

    Layer (type)                Output Shape  Param #  Connected to
    ====================================================================
    input (InputLayer)          [(None, 5)]   0        []
    dense_1 (Dense)             (None, 64)    384      ['input[0][0]']
    dense_2 (Dense)             (None, 32)    2080     ['dense_1[0][0]']
    dense_3 (Dense)             (None, 1)     33       ['dense_2[0][0]']
    dense_4 (Dense)             (None, 12)    396      ['dense_2[0][0]']
    concatenate (Concatenate)   (None, 13)    0        ['dense_3[0][0]',
                                                        'dense_4[0][0]']

    Parameters
    ----------
    input_length : int
        Number of predictors
    hpar_ls : dict[str, Any]
        Contains several hyperparameter
    dtype : str
        Should be either float32 or float64

    Returns
    -------
    Model
        _description_
    """
    tf.keras.backend.set_floatx(dtype)

    ### Build network ###
    # Input
    input = Input(shape=input_length, name="input", dtype=dtype)

    # Hidden layers
    hidden_1 = Dense(
        units=hpar_ls["lay1"], activation=hpar_ls["actv"], dtype=dtype
    )(input)
    hidden_2 = Dense(
        units=hpar_ls["lay1"] / 2, activation=hpar_ls["actv"], dtype=dtype
    )(hidden_1)

    # Different activation functions for output (alpha_0 and positive
    # increments)
    alpha0_out = Dense(units=1, dtype=dtype)(hidden_2)
    alphai_out = Dense(
        units=hpar_ls["p_degree"], activation="softplus", dtype=dtype
    )(hidden_2)

    # Concatenate output
    output = Concatenate()([alpha0_out, alphai_out])

    # Define model
    model = Model(inputs=input, outputs=output)

    # Return model
    return model
