from abc import ABC, abstractmethod
from typing import Any

from nptyping import NDArray
from tensorflow.keras import Model  # type: ignore


class BaseModel(ABC):
    """
    An abstract base class for creating the Neural Networks.

    Attributes
    ----------
    deep_arch : list
        A list containing layer types and number of neurons in each layer of
        the model.
    n_ens : int
        An integer representing the number of ensembles to be used in the
        model.
    n_cores : int
        An integer representing the number of cores to be used in the model.
    rpy_elements : dict
        A dictionary containing the R interface for evaluation.
    dtype : str
        A string representing the data type to be used in the model.
    hpar : dict
        A dictionary containing hyperparameters of the model.
    model : Model
        A reference to the underlying model instance.
    runtime_est : float
        A floating-point value representing the runtime for prediction.
    p_dropout : None or float
        A reference to the dropout probability, or None if not applicable.

    Methods
    -------
    _get_architecture(n_samples: int, n_features: int) -> Model
        An abstract method for creating the model architecture.
    _build(n_samples: int, n_features: int) -> Model
        An abstract method for building the model.
    fit(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray,
    y_valid: np.ndarray) -> None
        An abstract method for training the model.
    predict(X_test: np.ndarray) -> np.ndarray
        An abstract method for making predictions using the trained model.
    get_results(y_test: np.ndarray) -> dict
        An abstract method for computing the model's performance metrics.
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
        self.dataset = dataset
        self.ens_method = ens_method

    @abstractmethod
    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        """
        Abstract method for creating the architecture of the model.

        Parameters
        ----------
        n_samples : int
            An integer representing the number of samples in the dataset.
        n_features : int
            An integer representing the number of features in the dataset.

        Returns
        -------
        Model
            A reference to the underlying model instance.
        """
        pass

    @abstractmethod
    def _build(self, n_samples: int, n_features: int) -> Model:
        """
        Abstract method for building the model.

        Parameters
        ----------
        n_samples : int
            An integer representing the number of samples in the dataset.
        n_features : int
            An integer representing the number of features in the dataset.

        Returns
        -------
        Model
            A reference to the underlying model instance.
        """
        pass

    @abstractmethod
    def _get_loss(self):
        """
        Returns the loss function that is used in the model.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_valid: NDArray,
        y_valid: NDArray,
    ) -> None:
        """
        Abstract method for training the model.

        Parameters
        ----------
        X_train : np.ndarray
            A numpy array representing the training input data.
        y_train : np.ndarray
            A numpy array representing the training target data.
        X_valid : np.ndarray
            A numpy array representing the validation input data.
        y_valid : np.ndarray
            A numpy array representing the validation target data.
        """
        pass

    @abstractmethod
    def predict(self, X_test: NDArray) -> None:
        """
        Abstract method for making predictions using the trained model.

        Parameters
        ----------
        X_test : np.ndarray
            A numpy array representing the test input data.

        Returns
        -------
        np.ndarray
            A numpy array representing the model predictions for the given
            input data.
        """
        pass

    @abstractmethod
    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        """
        Abstract method for returning model evaluation metrics.

        Parameters
        ----------
        y_test : np.ndarray
            A numpy array representing the test target data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing evaluation metrics such as accuracy,
            precision, recall, f1 score, etc.
        """
        pass
