from abc import ABC, abstractmethod
from typing import Any

from nptyping import NDArray
from tensorflow.keras import Model  # type: ignore


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
        self.runtime_est = 0
        self.p_dropout = None

    @abstractmethod
    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        pass

    @abstractmethod
    def _build(self, n_samples: int, n_features: int) -> Model:
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
