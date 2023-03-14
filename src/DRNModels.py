import copy
import time
from typing import Any

import edward2 as ed
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from concretedropout.tensorflow import (ConcreteDenseDropout,
                                        get_dropout_regularizer,
                                        get_weight_regularizer)
from nptyping import Float, NDArray
from rpy2.robjects import vectors
from rpy2.robjects.conversion import localconverter
from scipy.special import logsumexp
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import Concatenate, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import L2  # type: ignore

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
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar = {
            "loss": ["tnorm", 0.95, 1],  # naval [0.95, 1], wine [0, 10]
            "lr_adam": 5e-4,  # -1 for Adam-default
            "n_epochs": 150,
            "n_patience": 10,
            "n_batch": 64,
            "lay1": 64,
            "actv": "softplus",
            "nn_verbose": 0,
            "run_eagerly": False,
        }

    def _build(self, n_samples: int, n_features: int) -> Model:
        # Custom optizimer
        if self.hpar["lr_adam"] == -1:
            custom_opt = "adam"
        else:
            custom_opt = Adam(learning_rate=self.hpar["lr_adam"])

        ### Build network ###
        model = self._get_architecture(
            n_samples=n_samples, n_features=n_features
        )

        # Get custom loss
        custom_loss = self._get_loss()

        ### Estimation ###
        # Compile model
        model.compile(
            optimizer=custom_opt,
            loss=custom_loss,
            run_eagerly=self.hpar["run_eagerly"],
        )

        # Return model
        return model

    def _get_loss(self):
        # Get standard logistic distribution
        tfd = tfp.distributions.Normal(
            loc=0,
            scale=1
            # loc=np.float64(0), scale=np.float64(1)
        )  # , dtype=tf.float64)

        # Lower and Upper bound
        loss, l, u = self.hpar["loss"]
        l_mass = K.constant(0)
        u_mass = K.constant(0)

        if loss == "0tnorm":
            l = 0  # noqa: E741
            u = np.Inf
        if (l is None) and (u is None):
            loss = "norm"

        l = K.constant(l)  # noqa: E741
        u = K.constant(u)

        # Custom loss function
        def crps_norm_loss(y_true, y_pred):
            """See Gneiting et al. (2005)"""
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

        def crps_0tnorm_loss(y_true, y_pred):
            """See Thorarinsdottir and Gneiting (2010)"""
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

            # Calculate p
            p = tfd.cdf(mu / sigma)

            # Calculate CRPS
            res = (
                sigma
                / K.square(p)
                * (
                    z_y * p * (2 * tfd.cdf(z_y) + p - 2)
                    + 2 * p * tfd.prob(z_y)
                    - (1 / np.sqrt(np.pi)) * tfd.cdf((mu * np.sqrt(2)) / sigma)
                )
            )

            res = K.mean(res)

            return res

        def crps_tnorm_loss(y_true, y_pred):
            """See http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html#GenNormal"""  # noqa: E501
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
            z_l = (l - mu) / sigma
            z_u = (u - mu) / sigma

            # Get z in bounds z_l and z_u
            z = K.clip(z_y, min_value=z_l, max_value=z_u)

            # Calculate CRPS by dividing formula into four lines
            # as done in reference (CRAN)
            line1 = (
                K.abs(z_y - z)
                + z_u * K.square(u_mass)
                - z_l * K.square(l_mass)
            )

            factor = (1 - l_mass - u_mass) / (tfd.cdf(z_u) - tfd.cdf(z_l))

            line2 = (
                factor
                * z
                * (
                    (
                        2 * tfd.cdf(z)
                        - (
                            (
                                (1 - 2 * l_mass) * tfd.cdf(z_u)
                                + (1 - 2 * u_mass) * tfd.cdf(z_l)
                            )
                        )
                    )
                    / (1 - l_mass - u_mass)
                )
            )

            line3 = factor * (
                2 * tfd.prob(z)
                - 2 * tfd.prob(z_u) * u_mass
                - 2 * tfd.prob(z_l) * l_mass
            )

            line4 = (
                K.square(factor)
                * (1 / np.sqrt(np.pi))
                * (tfd.cdf(z_u * np.sqrt(2)) - tfd.cdf(z_l * np.sqrt(2)))
            )

            # Put together different lines
            crps_standard_normal = line1 + line2 + line3 - line4
            res = sigma * crps_standard_normal

            # Calculate mean
            res = K.mean(res)

            return res

        available_losses = {
            "norm": crps_norm_loss,
            "0tnorm": crps_0tnorm_loss,
            "tnorm": crps_tnorm_loss,
        }

        return available_losses.get(loss)

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
        self.tr_scale[self.tr_scale == 0.0] = 1.0

        # Scale training data
        X_train = (X_train - self.tr_center) / self.tr_scale  # type: ignore

        # Scale validation data with training data attributes
        X_valid = (X_valid - self.tr_center) / self.tr_scale  # type: ignore

        ### Build model ###
        self.model = self._build(
            n_samples=X_train.shape[0], n_features=X_train.shape[1]
        )

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

        self._store_params()

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

    def _get_mu_activation_function(self):
        if self.hpar["loss"][0] == "0tnorm":
            return "softplus"
        elif self.hpar["loss"][0] == "tnorm":

            def truncated_activation(x):
                _, l, u = self.hpar["loss"]
                return tf.keras.activations.sigmoid(x) * (u - l) + l

            return truncated_activation

        else:
            return "linear"

    def _store_params(self):
        pass


class DRNRandInitModel(DRNBaseModel):
    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
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
        input = Input(shape=n_features, name="input", dtype=self.dtype)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Build layers
            if idx == 0:
                hidden_layer = Dense(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = Dense(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = Dense(units=1, activation=activation_mu, dtype=self.dtype)(
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
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "training": True,
                "p_dropout": 0.05,
                "p_dropout_input": 0,
                "upscale_units": True,
                "n_batch": 64,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
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
        p_dropout: float = self.hpar["p_dropout"]

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)
        # Input dropout
        input_d = Dropout(
            rate=self.hpar["p_dropout_input"], noise_shape=(n_features,)
        )(input, training=self.hpar["training"])

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Calculate units
            if self.hpar["upscale_units"]:
                n_units = int(layer_info[1] / (1 - p_dropout))
            else:
                n_units = layer_info[1]
            # Build layers
            if idx == 0:
                hidden_layer = Dense(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input_d)
                hidden_d_layer = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])
            else:
                hidden_layer = Dense(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_d_layer  # type: ignore
                )
                hidden_d_layer = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = Dense(units=1, activation=activation_mu, dtype=self.dtype)(
            hidden_layer  # type: ignore
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


class DRNBayesianModel(DRNBaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "n_epochs": 500,
                "prior": "standard_normal",
                "posterior": "mean_field",
                "post_scale_scaling": 0.001,
            }
        )

    # def _get_loss(self):
    #     def negloglik(y_true, y_pred):
    #         dist = tfp.distributions.Normal(loc=y_pred[0], scale=y_pred[1])
    #         return K.sum(-dist.log_prob(y_true))

    #     return negloglik

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Get prior and posterior
        prior_fn = self._get_prior()
        posterior_fn = self._get_posterior()

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Build layers
            if idx == 0:
                hidden_layer = tfp.layers.DenseVariational(
                    units=layer_info[1],
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / n_samples,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    input
                )  # TODO: Is training parameter needed?
            else:
                hidden_layer = tfp.layers.DenseVariational(
                    units=layer_info[1],
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / n_samples,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / n_samples,
            dtype=self.dtype,
            activation=activation_mu,
        )(
            hidden_layer  # type: ignore
        )
        scale_out = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / n_samples,
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

        available_priors = {
            "standard_normal": prior_standard_normal,
            "uniform": prior_uniform,
            "laplace": prior_laplace,
        }

        return available_priors.get(self.hpar["prior"])

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
                    tfp.layers.VariableLayer(2 * n),
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Normal(
                                loc=t[..., :n],
                                scale=1e-5
                                + self.hpar["post_scale_scaling"]
                                * tf.nn.softplus(c + t[..., n:]),
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    ),
                ]
            )
            return posterior_model

        return posterior_mean_field


class DRNVariationalDropoutModel(DRNBaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "n_epochs": 150,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Build layers
            if idx == 0:
                hidden_layer = ed.layers.DenseVariationalDropout(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input, training=False)
            else:
                hidden_layer = ed.layers.DenseVariationalDropout(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer, training=False  # type: ignore
                )

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = Dense(units=1, activation=activation_mu, dtype=self.dtype)(
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


class DRNConcreteDropoutModel(DRNBaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "tau": 1.0,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        wr = get_weight_regularizer(N=n_samples, l=1e-2, tau=self.hpar["tau"])
        dr = get_dropout_regularizer(
            N=n_samples, tau=self.hpar["tau"], cross_entropy_loss=False
        )
        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Build layers
            if idx == 0:
                hidden_layer = ConcreteDenseDropout(
                    Dense(
                        units=layer_info[1],
                        activation=self.hpar["actv"],
                        dtype=self.dtype,
                    ),
                    weight_regularizer=wr,
                    dropout_regularizer=dr,
                    is_mc_dropout=True,
                )(input)
            else:
                hidden_layer = ConcreteDenseDropout(
                    Dense(
                        units=layer_info[1],
                        activation=self.hpar["actv"],
                        dtype=self.dtype,
                    ),
                    weight_regularizer=wr,
                    dropout_regularizer=dr,
                    is_mc_dropout=True,
                )(
                    hidden_layer  # type: ignore
                )

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = Dense(units=1, activation=activation_mu, dtype=self.dtype)(
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

    def _store_params(self):
        if self.p_dropout is None:
            self.p_dropout = []
        for layer in self.model.layers:
            if isinstance(layer, ConcreteDenseDropout):
                self.p_dropout.append(
                    tf.nn.sigmoid(layer.trainable_variables[0]).numpy()[0]
                )
        print(f"Learned Dropout rates: {self.p_dropout}")


class DRNBatchEnsembleModel(DRNBaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "n_epochs": 500,
                "n_batch": 60,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Calculate batch size ###

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Make initializer
        def make_initializer(num):
            return ed.initializers.RandomSign(num)

        # Hidden layers
        for idx, layer_info in enumerate(self.deep_arch):
            # Build layers
            if idx == 0:
                hidden_layer = ed.layers.DenseBatchEnsemble(
                    units=layer_info[1],
                    rank=1,
                    ensemble_size=self.n_ens,
                    use_bias=True,
                    alpha_initializer=make_initializer(0.5),  # type: ignore
                    gamma_initializer=make_initializer(0.5),  # type: ignore
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = ed.layers.DenseBatchEnsemble(
                    units=layer_info[1],
                    rank=1,
                    ensemble_size=self.n_ens,
                    use_bias=True,
                    alpha_initializer=make_initializer(0.5),  # type: ignore
                    gamma_initializer=make_initializer(0.5),  # type: ignore
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Different activation functions for output
        activation_mu = self._get_mu_activation_function()
        loc_out = ed.layers.DenseBatchEnsemble(
            units=1,
            rank=1,
            ensemble_size=self.n_ens,
            use_bias=True,
            alpha_initializer=make_initializer(0.5),  # type: ignore
            gamma_initializer=make_initializer(0.5),  # type: ignore
            dtype=self.dtype,
            activation=activation_mu,
        )(
            hidden_layer  # type: ignore
        )
        scale_out = ed.layers.DenseBatchEnsemble(
            units=1,
            rank=1,
            ensemble_size=self.n_ens,
            use_bias=True,
            alpha_initializer=make_initializer(0.5),  # type: ignore
            gamma_initializer=make_initializer(0.5),  # type: ignore
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

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale  # type: ignore

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Extract all trained weights
        weights = self.model.get_weights()
        # Copy weights to adjust them for the ensemble members
        new_weights = copy.deepcopy(weights)
        # Create new model with same architecture but n_ens=1
        new_model = self._build_single_model(
            n_samples=X_pred.shape[0], n_features=X_pred.shape[1]
        )

        # Initialize predictions
        self.predictions = []

        # Iterate and extract each ensemble member
        for i_ens in range(self.n_ens):

            # Iterate over layers and extract new model's weights
            for i_layer_weights, layer_weights in enumerate(weights):
                # Keep shared weights
                if (i_layer_weights % 4) == 0:
                    new_weights[i_layer_weights] = layer_weights
                # Extract alphas, gammas and bias
                elif (i_layer_weights % 4) != 0:
                    new_weights[i_layer_weights] = np.reshape(
                        layer_weights[i_ens],
                        newshape=(1, layer_weights.shape[1]),
                    )

            # Set new weights
            new_model.set_weights(new_weights)

            # Make predictions with temporary models
            # In order to match dimensions in DenseBatchEnsemble use batchsize
            # from training
            self.predictions.append(
                new_model.predict(
                    X_pred,
                    verbose=self.hpar["nn_verbose"],
                    batch_size=self.hpar["n_batch"],
                )
            )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def _build_single_model(self, n_samples, n_features):
        # Save originial n_ens
        n_ens_original = self.n_ens

        # Use n_ens = 1 for temporary model
        self.n_ens = 1
        model = self._build(n_samples=n_samples, n_features=n_features)

        # Reset n_ens to original value
        self.n_ens = n_ens_original

        return model

    def get_results(self, y_test: NDArray) -> list[dict[str, Any]]:
        # Initialize results
        results = []

        ### Evaluation ###
        # Calculate evaluation measres of DRN forecasts
        for f in self.predictions:
            scores = fn_scores_distr(
                f=f, y=y_test, distr="norm", rpy_elements=self.rpy_elements
            )

            results.append(
                {
                    "f": f,
                    "nn_ls": self.hpar,
                    "scores": scores,
                    "n_train": self.n_train,
                    "n_valid": self.n_valid,
                    "n_test": self.n_test,
                    "runtime_est": self.runtime_est,
                    "runtime_pred": self.runtime_pred,
                }
            )

        ### Output ###
        # Output
        return results


class DRNEvaModel(DRNBaseModel):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch, n_ens, n_cores, rpy_elements, dtype, **kwargs
        )
        self.hpar.update(
            {
                "training": True,
                "p_dropout": kwargs["p_dropout"],
                "p_dropout_input": 0,
                "upscale_units": True,
                "n_batch": 128,
                "n_epochs": 4_000,
                "actv": "relu",
                "tau": kwargs["tau"],
                "adam": -1,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        # Get dropout rate
        p_dropout = self.hpar["p_dropout"]

        # Initialize L2 regularizer
        lengthscale = 1e-2
        reg = (
            lengthscale**2
            * (1 - p_dropout)
            / (2.0 * n_samples * self.hpar["tau"])
        )

        ### Build network ###
        # Input
        input = Input(shape=n_features, name="input", dtype=self.dtype)
        x = Dropout(rate=p_dropout, noise_shape=(n_features,))(
            input, training=self.hpar["training"]
        )
        x = Dense(
            units=50,
            activation=self.hpar["actv"],
            kernel_regularizer=L2(l2=reg),
            dtype=self.dtype,
        )(x)
        x = Dropout(rate=p_dropout, noise_shape=(50,))(
            x, training=self.hpar["training"]
        )

        # Model output
        output = Dense(units=1, kernel_regularizer=L2(l2=reg))(x)

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def _get_loss(self):
        return "mean_squared_error"

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
        self.tr_scale[self.tr_scale == 0.0] = 1.0

        # Scale training data
        X_train = (X_train - self.tr_center) / self.tr_scale  # type: ignore

        # Scale validation data with training data attributes
        X_valid = (X_valid - self.tr_center) / self.tr_scale  # type: ignore

        # Scale y train
        self.y_center = np.mean(y_train, axis=0)
        self.y_scale = np.std(y_train, axis=0)
        if self.y_scale == 0:
            self.y_scale = 1
        y_train_normalized = (y_train - self.y_center) / self.y_scale

        ### Build model ###
        self.model = self._build(
            n_samples=X_train.shape[0], n_features=X_train.shape[1]
        )

        # Take time
        start_tm = time.time_ns()

        ### Fit model ###
        self.model.fit(
            x=X_train,
            y=y_train_normalized,
            epochs=self.hpar["n_epochs"],
            batch_size=self.hpar["n_batch"],
            verbose=self.hpar["nn_verbose"],
        )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_est = end_tm - start_tm

        self._store_params()

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale  # type: ignore

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Predict 10_000 times on scaled data
        self.f: NDArray[Any, Float] = np.array(
            [
                self.model.predict(
                    X_pred, batch_size=500, verbose=self.hpar["nn_verbose"]
                )
                for _ in range(10_000)
            ]
        )

        self.f = self.f * self.y_scale + self.y_center
        self.mc_pred = np.mean(self.f, axis=0)

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        ### Evaluation ###
        # Calculate evaluation measres of DRN forecasts
        scoring_rules = self.rpy_elements["scoring_rules"]
        np_cv_rules = self.rpy_elements["np_cv_rules"]

        y_vector = vectors.FloatVector(y_test)

        with localconverter(np_cv_rules) as cv:  # noqa: F841
            crps = np.mean(
                scoring_rules.crps_sample(y=y_vector, dat=self.mc_pred)
            )

        crps_eva = crps_mixnorm_mc(
            self.mc_pred.reshape((1, self.mc_pred.shape[0])), y_test, 1
        )

        rmse = (
            np.mean((y_test.squeeze() - self.mc_pred.squeeze()) ** 2.0) ** 0.5
        )

        ll = (
            logsumexp(
                -0.5 * self.hpar["tau"] * (y_test[None] - self.f) ** 2.0, 0
            )
            - np.log(10_000)
            - 0.5 * np.log(2 * np.pi)
            + 0.5 * np.log(self.hpar["tau"])
        )
        test_ll = np.mean(ll)

        ### Output ###
        # Output
        return {
            "f": self.f,
            "rmse": rmse,
            "crps": crps,
            "crps_eva": crps_eva,
            "logl": test_ll,
            "nn_ls": self.hpar,
            # "scores": scores,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "runtime_est": self.runtime_est,
            "runtime_pred": self.runtime_pred,
        }
