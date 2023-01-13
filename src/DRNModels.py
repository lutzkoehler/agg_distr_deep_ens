import time
from typing import Any

import edward2 as ed
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# from ddrop.layers import DropConnect
from dropconnect_tensorflow import DropConnectDense
from nptyping import Float, NDArray
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import Concatenate, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from BaseModel import BaseModel
from fn_eval import fn_scores_distr


class DRNBaseModel(BaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar = {
            "lr_adam": 5e-4,  # -1 for Adam-default
            "n_epochs": 150,
            "n_patience": 10,
            "n_batch": 64,
            "lay1": 64,
            "actv": "softplus",
            "nn_verbose": 0,
            "p_dropout": 0.1,
            "p_dropout_input": 0.2,
        }

    def _build(self, input_length: int) -> Model:
        # Custom optizimer
        if self.hpar["lr_adam"] == -1:
            custom_opt = "adam"
        else:
            custom_opt = Adam(learning_rate=self.hpar["lr_adam"])

        ### Build network ###
        model = self._get_architecture(
            input_length=input_length, training=self.training
        )

        # Get custom loss
        custom_loss = self._get_loss()

        ### Estimation ###
        # Compile model
        model.compile(optimizer=custom_opt, loss=custom_loss)

        # Return model
        return model

    def _get_loss(self):
        # Get standard logistic distribution
        tfd = tfp.distributions.Normal(
            loc=0,
            scale=1
            # loc=np.float64(0), scale=np.float64(1)
        )  # , dtype=tf.float64)

        # Custom loss function
        def custom_loss(y_true, y_pred):
            # Get location and scale
            mu = K.dot(
                y_pred,
                K.constant(np.r_[1, 0], shape=(2, 1)),  # , dtype=np.float64)
            )
            sigma = K.dot(
                y_pred,
                K.constant(np.r_[0, 1], shape=(2, 1)),  # , dtype=np.float64)
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

        return custom_loss

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_valid: NDArray,
        y_valid: NDArray,
    ) -> None:
        ### Data preparation ###
        # Read out set sizes
        self.n_train = X_train.shape[0]
        self.n_valid = X_valid.shape[0]

        # Save center and scale parameters
        self.tr_center: NDArray = np.mean(X_train, axis=0)
        self.tr_scale: NDArray = np.std(X_train, axis=0)

        # Scale training data
        X_train = (X_train - self.tr_center) / self.tr_scale  # type: ignore

        # Scale validation data with training data attributes
        X_valid = (X_valid - self.tr_center) / self.tr_scale  # type: ignore

        ### Build model ###
        self.model = self._build(input_length=X_train.shape[1])

        # Take time
        start_tm = time.time_ns()

        ### Fit model ###
        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=self.hpar["n_epochs"],
            batch_size=self.hpar["n_batch"],
            validation_data=(X_valid, y_valid),
            verbose=self.hpar["nn_verbose"],
            callbacks=EarlyStopping(
                patience=self.hpar["n_patience"],
                restore_best_weights=True,
                monitor="val_loss",
            ),
        )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_est = end_tm - start_tm

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale  # type: ignore

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Predict parameters of distributional forecasts (on scaled data)
        self.f: NDArray[Any, Float] = self.model.predict(
            X_pred, verbose=self.hpar["nn_verbose"]
        )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        ### Evaluation ###
        # Calculate evaluation measres of DRN forecasts
        scores = fn_scores_distr(
            f=self.f, y=y_test, distr="norm", rpy_elements=self.rpy_elements
        )

        ### Output ###
        # Output
        return {
            "f": self.f,
            "nn_ls": self.hpar,
            "scores": scores,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "runtime_est": self.runtime_est,
            "runtime_pred": self.runtime_pred,
        }


class DRNRandInitModel(DRNBaseModel):
    def _get_architecture(
        self, input_length: int, training: bool = False
    ) -> Model:
        """Construct and return DRN base model

        Architecture:

        Layer (type)                Output Shape    Param #  Connected to
        =========================================================================
        input (InputLayer)          [(None, 5)]     0        []
        dense_1 (Dense)             (None, 64)      384      ['input[0][0]']
        dense_2 (Dense)             (None, 32)      2080     ['dense_1[0][0]']
        dense_3 (Dense)             (None, 1)       33       ['dense_2[0][0]']
        dense_4 (Dense)             (None, 1)       33       ['dense_2[0][0]']
        concatenate (Concatenate)   (None, 2)       0        ['dense_3[0][0]',
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
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=input_length, name="input", dtype=self.dtype)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Get layer class
            if layer_info[0] == "Dense":
                layer_class = Dense
            else:
                layer_class = Dense
            # Build layers
            if idx == 0:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # hidden_1 = Dense(
        #     units=self.hpar["lay1"],
        #     activation=self.hpar["actv"],
        #     dtype=self.dtype,
        # )(input)
        # hidden_2 = Dense(
        #     units=self.hpar["lay1"] / 2,
        #     activation=self.hpar["actv"],
        #     dtype=self.dtype,
        # )(hidden_1)

        # Different activation functions for output
        loc_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        scale_out = Dense(units=1, activation="softplus", dtype=self.dtype)(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([loc_out, scale_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class DRNDropoutModel(DRNBaseModel):
    def _get_architecture(
        self, input_length: int, training: bool = False
    ) -> Model:
        """Construct and return DRN base model

        Architecture:

        Layer (type)                Output Shape    Param # Connected to
        =======================================================================
        input (InputLayer)          [(None, 5)]     0       []
        dropout (Dropout)           (None, 5)       0       ['input[0][0]']
        dense (Dense)               (None, 128)     768     ['dropout[0][0]']
        dropout_1 (Dropout)         (None, 128)     0       ['dense[0][0]']
        dense_1 (Dense)             (None, 64)      8256    ['dropout_1[0][0]']
        dropout_2 (Dropout)         (None, 64)      0       ['dense_1[0][0]']
        dense_2 (Dense)             (None, 1)       65      ['dropout_2[0][0]']
        dense_3 (Dense)             (None, 1)       65      ['dropout_2[0][0]']
        concatenate (Concatenate)   (None, 2)       0       ['dense_2[0][0]',
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
        tf.keras.backend.set_floatx(self.dtype)

        # Extract params
        p_dropout = self.hpar["p_dropout"]

        ### Build network ###
        # Input
        input = Input(shape=(input_length,), name="input", dtype=self.dtype)
        # Input dropout
        # input_d = Dropout(
        #     rate=self.hpar["p_dropout_input"], noise_shape=(input_length,)
        # )(input, training=training)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Get layer class
            if layer_info[0] == "Dense":
                layer_class = Dense
            else:
                layer_class = Dense
            # Calculate units
            n_units = int(layer_info[1] / (1 - p_dropout))
            # Build layers
            if idx == 0:
                hidden_layer = layer_class(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
                hidden_d_layer = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=training)
            else:
                hidden_layer = layer_class(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_d_layer  # type: ignore
                )
                hidden_d_layer = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=training)

        # Different activation functions for output
        loc_out = Dense(units=1, dtype=self.dtype)(
            hidden_d_layer  # type: ignore
        )
        scale_out = Dense(units=1, activation="softplus", dtype=self.dtype)(
            hidden_d_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([loc_out, scale_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # print(self.deep_arch)
        # Return model
        return model


class DRNDropConnectModel(DRNBaseModel):
    def _get_architecture(self, input_length: int, training: bool) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        # Extract params
        p_dropout = self.hpar["p_dropout"]

        ### Build network ###
        # Input
        input = Input(shape=(input_length,), name="input", dtype=self.dtype)
        # Input dropout
        # input_d = Dropout(
        #     rate=self.hpar["p_dropout_input"], noise_shape=(input_length,)
        # )(input, training=training)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Get layer class
            # if layer_info[0] == "Dense":
            #     layer_class = DropConnectDense
            # else:
            #     layer_class = DropConnectDense
            # Calculate units
            # n_units = int(layer_info[1] / (1 - p_dropout))
            # Build layers
            if idx == 0:
                hidden_layer = DropConnectDense(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                    prob=p_dropout,
                    use_bias=True,
                )(input, training=training)
            else:
                hidden_layer = DropConnectDense(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                    prob=p_dropout,
                    use_bias=True,
                )(
                    hidden_layer, training=training  # type: ignore
                )

        # Different activation functions for output
        loc_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        scale_out = Dense(units=1, activation="softplus", dtype=self.dtype)(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([loc_out, scale_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class DRNBayesianModel(DRNBaseModel):
    # def _get_loss(self):
    #     def negloglik(y_true, y_pred):
    #         dist = tfp.distributions.Normal(loc=y_pred[0], scale=y_pred[1])
    #         return K.sum(-dist.log_prob(y_true))

    #     return negloglik

    def _get_architecture(self, input_length: int, training: bool) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(input_length,), name="input", dtype=self.dtype)

        # Get prior and posterior
        prior_fn = self._get_prior()
        posterior_fn = self._get_posterior()

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Get layer class
            if layer_info[0] == "Dense":
                layer_class = tfp.layers.DenseVariational
            else:
                layer_class = tfp.layers.DenseVariational
            # Build layers
            if idx == 0:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / 5000,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input, training=training)
            else:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / 5000,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer, training=training  # type: ignore
                )

        # Different activation functions for output
        loc_out = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / 5000,
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )
        scale_out = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / 5000,
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([loc_out, scale_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def _get_prior(self):
        """Returns the prior weight distribution

        e.g. Normal of mean=0 and sd=1
        Prior might be trainable or not

        Returns
        -------
        function
            prior function
        """

        def prior_standard_normal(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.MultivariateNormalDiag(
                            loc=tf.zeros(n), scale_diag=tf.ones(n)
                        )
                    )
                ]
            )
            return prior_model

        def prior_uniform(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Uniform(
                                low=tf.ones(n) * -3, high=tf.ones(n) * 3
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    )
                ]
            )
            return prior_model

        def prior_laplace(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Laplace(
                                loc=tf.zeros(n), scale=tf.ones(n)
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    )
                ]
            )
            return prior_model

        return prior_uniform

    def _get_posterior(self):
        """Returns the posterior weight distribution

        e.g. multivariate Gaussian
        Depending on the distribution the learnable parameters vary

        Returns
        -------
        function
            posterior function
        """

        def posterior_mean_field(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.0))
            posterior_model = Sequential(
                [
                    # tfp.layers.VariableLayer(
                    #     tfp.layers.MultivariateNormalTriL.params_size(n),
                    #     dtype=dtype,
                    # ),
                    # tfp.layers.MultivariateNormalTriL(n)
                    tfp.layers.VariableLayer(2 * n),
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Normal(
                                loc=t[..., :n],
                                scale=1e-5
                                + 1 * tf.nn.softplus(c + t[..., n:]),
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    ),
                ]
            )
            return posterior_model

        return posterior_mean_field


class DRNVariationalDropoutModel(DRNBaseModel):
    def _get_architecture(self, input_length: int, training: bool) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(input_length,), name="input", dtype=self.dtype)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Get layer class
            if layer_info[0] == "Dense":
                layer_class = ed.layers.DenseVariationalDropout
            else:
                layer_class = ed.layers.DenseVariationalDropout
            # Build layers
            if idx == 0:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input, training=training)
            else:
                hidden_layer = layer_class(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer, training=training  # type: ignore
                )

        # Different activation functions for output
        loc_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        scale_out = Dense(units=1, activation="softplus", dtype=self.dtype)(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([loc_out, scale_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model
