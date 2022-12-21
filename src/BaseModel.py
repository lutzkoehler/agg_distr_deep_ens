from abc import ABC, abstractmethod
from typing import Any

from nptyping import NDArray
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.layers import Dropout, InputLayer  # type: ignore


class BaseModel(ABC):
    def __init__(
        self,
        nn_deep_arch: list[Any],
        n_ens: int,
        n_cores: int,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        **kwargs
    ) -> None:
        self.deep_arch = nn_deep_arch
        self.n_ens = n_ens
        self.n_cores = n_cores
        self.rpy_elements = rpy_elements
        self.dtype = dtype
        self.hpar = {}
        self.model: Model

    @abstractmethod
    def _get_architecture(self, input_length: int, training: bool) -> Model:
        pass

    @abstractmethod
    def _build(self, input_length: int) -> Model:
        pass

    @abstractmethod
    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_valid: NDArray,
        y_valid: NDArray,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: NDArray) -> None:
        pass

    @abstractmethod
    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        pass


class DropoutBaseModel(BaseModel):
    def scale_weights(self) -> None:
        """Get weights affected by dropout layer and scale down by p_dropout"""
        # Iterate over layers and check if following layer is dropout
        # If yes, adjust by p_dropout (p_dropout_input for input layer)
        layers = self.model.layers
        for idx, layer in enumerate(layers):
            if idx + 1 == len(layers):
                break
            if not isinstance(layers[idx], InputLayer) and isinstance(
                layers[idx + 1], Dropout
            ):
                new_weights = [layer.get_weights()[0] * self.hpar["p_dropout"]]
                if len(layer.get_weights()) > 1:
                    new_weights.append(layer.get_weights()[1])
                layer.set_weights(new_weights)
