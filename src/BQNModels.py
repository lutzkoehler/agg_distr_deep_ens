import copy
import logging
import time
from typing import Any

import edward2 as ed
import keras.backend as K
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import tensorflow_probability as tfp
from concretedropout.tensorflow import (ConcreteDenseDropout,
                                        get_dropout_regularizer,
                                        get_weight_regularizer)
from nptyping import Float, NDArray
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import Concatenate, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from BaseModel import BaseModel
from fn_eval import bern_quants, fn_scores_ens

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BQNBaseModel(BaseModel):
    """
    Represents the base class for Bernstein Quantile Networks.
    For a documentation of the methods look at BaseModel
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            **kwargs,
        )
        # If not given use equidistant quantiles (multiple of ensemble
        # coverage, incl. median)
        if q_levels is None:
            step = 1 / 100
            self.q_levels = np.arange(start=1 / 100, stop=1 + step, step=step)
        else:
            self.q_levels = q_levels
        self.hpar = {
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
            "run_eagerly": False,
        }

    def _build(self, n_samples: int, n_features: int) -> Model:
        # Calculate equidistant quantile levels for loss function
        self.q_levels_loss = np.arange(
            start=1 / (self.hpar["n_q"] + 1),
            stop=(self.hpar["n_q"] + 1) / (self.hpar["n_q"] + 1),
            step=1 / (self.hpar["n_q"] + 1),
        )

        # Basis of Bernstein polynomials evaluated at quantile levels
        self.B = np.apply_along_axis(
            func1d=ss.binom.pmf,
            arr=np.reshape(
                np.arange(start=0, stop=self.hpar["p_degree"] + 1, step=1),
                newshape=(1, self.hpar["p_degree"] + 1),
            ),
            axis=0,
            p=self.q_levels_loss,
            n=self.hpar["p_degree"],
        )

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
        # Quantile loss functions (for neural network)
        def qt_loss(y_true, y_pred):
            """Quantile loss for BQN network

            Quantile Loss:
            L(yp_i, y_i) = max[ q(y_i - yp_i), (q - 1)(y_i - yp_i) ]

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
            q = K.dot(
                K.cumsum(y_pred, axis=1), K.constant(self.B.T)
            )  # , dtype=np.float64))

            # Calculate individual quantile scores
            err = y_true - q  # err_shape (64,99)
            e1 = err * K.constant(
                value=self.q_levels_loss,
                shape=(1, self.hpar["n_q"]),  # , dtype=np.float64
            )  # e1_shape (64,99)
            e2 = err * K.constant(
                value=self.q_levels_loss - 1,
                shape=(1, self.hpar["n_q"]),  # , dtype=np.float64
            )

            # Find correct values (max) and return mean
            return K.mean(
                K.maximum(e1, e2), axis=1
            )  # max_shape (64,99) - mean_shape (64,)

        return qt_loss

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
        self.tr_center = np.mean(X_train, axis=0)
        self.tr_scale = np.std(X_train, axis=0)
        self.tr_scale[self.tr_scale == 0.0] = 1.0

        # Scale training data
        X_train = (X_train - self.tr_center) / self.tr_scale

        # Scale validation data with training data attributes
        X_valid = (X_valid - self.tr_center) / self.tr_scale

        ### Build model ###
        self.model = self._build(
            n_samples=X_train.shape[0], n_features=X_train.shape[1]
        )

        # Take time
        start_tm = time.time_ns()

        # Fit model
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
        X_pred = (X_test - self.tr_center) / self.tr_scale

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Predict coefficients of Bernstein polynomials
        n_mean_prediction = self.hpar.get("n_mean_prediction")
        if n_mean_prediction is None:
            self.coeff_bern = self.model.predict(
                X_pred, verbose=self.hpar["nn_verbose"]
            )
        else:
            # Predict n_mean_prediction times if single model
            mc_pred: NDArray[Any, Float] = np.array(
                [
                    self.model.predict(
                        X_pred, batch_size=500, verbose=self.hpar["nn_verbose"]
                    )
                    for _ in range(n_mean_prediction)
                ]
            )
            self.coeff_bern = np.mean(mc_pred, axis=0)

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        # Accumulate increments
        coeff_bern = np.cumsum(self.coeff_bern, axis=1)

        ### Evaluation ###
        # Sum up calcuated quantiles (Sum of basis at quantiles times
        # coefficients)
        q = bern_quants(alpha=coeff_bern, q_levels=self.q_levels)
        # Calculate evaluation measres of DRN forecasts
        scores = fn_scores_ens(
            ens=q,
            y=y_test,
            skip_evals=["e_me"],
            scores_ens=True,
            rpy_elements=self.rpy_elements,
        )

        # Transform ranks to n_(ens+1) bins (for multiples of (n_ens+1) exact)
        if q.shape[1] != self.n_ens:
            scores["rank"] = np.ceil(
                scores["rank"] * (self.n_ens + 1) / (q.shape[1] + 1)
            )

        # Calculate bias of mean forecast (formula given)
        scores["e_me"] = np.mean(coeff_bern, axis=1) - y_test

        ### Output ###
        # Output
        return {
            "f": q,
            "alpha": coeff_bern,
            "nn_ls": self.hpar,
            "scores": scores,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "runtime_est": self.runtime_est,
            "runtime_pred": self.runtime_pred,
        }

    def _store_params(self):
        pass


class BQNRandInitModel(BQNBaseModel):
    """
    Class represents the naive ensemble method for BQNs.
    """

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
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

        # Different activation functions for output (alpha_0 and positive
        # increments)
        alpha0_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        alphai_out = Dense(
            units=self.hpar["p_degree"],
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([alpha0_out, alphai_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class BQNDropoutModel(BQNBaseModel):
    """
    Class represents the MC dropout method for BQNs.
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            q_levels,
            **kwargs,
        )
        self.hpar.update(
            {
                "training": True,
                "p_dropout": 0.05,
                "p_dropout_input": 0,
                "upscale_units": True,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        # Extract params
        p_dropout = self.hpar["p_dropout"]

        ### Build network ###
        # Input
        input = Input(shape=n_features, name="input", dtype=self.dtype)
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
                hidden_layer_d = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])
            else:
                hidden_layer = Dense(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer_d  # type: ignore
                )
                hidden_layer_d = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])

        # Different activation functions for output (alpha_0 and positive
        # increments)
        alpha0_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer_d  # type: ignore
        )
        alphai_out = Dense(
            units=self.hpar["p_degree"],
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer_d  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([alpha0_out, alphai_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class BQNBayesianModel(BQNBaseModel):
    """
    Class represents the Bayesian NN method for BQNs.
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            q_levels,
            **kwargs,
        )
        self.hpar.update(
            {
                "n_epochs": 500,
                "prior": "standard_normal",
                "posterior": "mean_field",
                "post_scale_scaling": 0.001,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )

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
                )(input)
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
        alpha0_out = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / n_samples,
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )
        alphai_out = tfp.layers.DenseVariational(
            units=self.hpar["p_degree"],
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / n_samples,
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([alpha0_out, alphai_out])

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


class BQNVariationalDropoutModel(BQNBaseModel):
    """
    Class represents the variational dropout method for BQNs.
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            rpy_elements,
            dtype,
            q_levels,
            **kwargs,
        )
        self.hpar.update(
            {
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
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
                )(input)
            else:
                hidden_layer = ed.layers.DenseVariationalDropout(
                    units=layer_info[1],
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Different activation functions for output (alpha_0 and positive
        # increments)
        alpha0_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        alphai_out = Dense(
            units=self.hpar["p_degree"],
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([alpha0_out, alphai_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class BQNConcreteDropoutModel(BQNBaseModel):
    """
    Class represents the concrete dropout method for BQNs.
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            q_levels,
            **kwargs,
        )
        self.hpar.update(
            {
                "tau": 1.0,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
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

        # Different activation functions for output (alpha_0 and positive
        # increments)
        alpha0_out = Dense(units=1, dtype=self.dtype)(
            hidden_layer  # type: ignore
        )
        alphai_out = Dense(
            units=self.hpar["p_degree"],
            activation="softplus",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Concatenate output
        output = Concatenate()([alpha0_out, alphai_out])

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
        with open(
            f"concrete_dropout_rates_{self.dataset}_{self.ens_method}.txt", "a"
        ) as myfile:
            myfile.write("BQN - Dropout_Rates: " + repr(self.p_dropout))
        log_message = f"Learned Dropout rates: {self.p_dropout}"
        logging.info(log_message)


class BQNBatchEnsembleModel(BQNBaseModel):
    """
    Class represents the BatchEnsemble method for BQNs.
    """

    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        q_levels: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            nn_deep_arch,
            n_ens,
            n_cores,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            q_levels,
            **kwargs,
        )
        self.hpar.update(
            {
                "n_epochs": 500,
                "n_batch": 60 * 1,
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

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

        # Different activation functions for output (alpha_0 and positive
        # increments)
        alpha0_out = ed.layers.DenseBatchEnsemble(
            units=1,
            rank=1,
            ensemble_size=self.n_ens,
            use_bias=True,
            alpha_initializer=make_initializer(0.5),  # type: ignore
            gamma_initializer=make_initializer(0.5),  # type: ignore
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )
        alphai_out = ed.layers.DenseBatchEnsemble(
            units=self.hpar["p_degree"],
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
        output = Concatenate()([alpha0_out, alphai_out])

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale

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
                # Extract alpha, gammas and bias
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
        for coeff_bern in self.predictions:
            # Accumulate increments
            coeff_bern = np.cumsum(coeff_bern, axis=1)

            ### Evaluation ###
            # Sum up calcuated quantiles (Sum of basis at quantiles times
            # coefficients)
            q = bern_quants(alpha=coeff_bern, q_levels=self.q_levels)
            # Calculate evaluation measres of DRN forecasts
            scores = fn_scores_ens(
                ens=q,
                y=y_test,
                skip_evals=["e_me"],
                scores_ens=True,
                rpy_elements=self.rpy_elements,
            )

            # Transform ranks to n_(ens+1) bins
            # (for multiples of (n_ens+1) exact)
            if q.shape[1] != self.n_ens:
                scores["rank"] = np.ceil(
                    scores["rank"] * (self.n_ens + 1) / (q.shape[1] + 1)
                )

            # Calculate bias of mean forecast (formula given)
            scores["e_me"] = np.mean(coeff_bern, axis=1) - y_test

            results.append(
                {
                    "f": q,
                    "alpha": coeff_bern,
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
