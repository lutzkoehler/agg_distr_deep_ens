import keras.backend as K
import numpy as np
import scipy.stats as ss

nn_ls = {
    "n_q": 99,
    "p_degree": 12,
}

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


def qt_loss(y_true, y_pred):
    # Quantiles calculated via basis and increments
    q = np.dot(
        np.cumsum(y_pred, axis=0),
        np.reshape(
            a=B.flatten(order="F"),
            newshape=(nn_ls["p_degree"] + 1, nn_ls["n_q"]),
        ),
    )
    q = np.dot(y_pred, B.T)
    print(q.shape)

    # Calculate individual quantile scores
    err = y_true[:, None] - q
    print(err.shape)
    e1 = err * np.reshape(a=q_levels_loss, newshape=(1, nn_ls["n_q"]))
    e2 = err * np.reshape(a=q_levels_loss - 1, newshape=(1, nn_ls["n_q"]))

    # Find corret values (max) and return mean
    # TODO: Check axis
    return np.mean(np.maximum(e1, e2), axis=1)


n = 1000
y_true = np.ones(n)
y_pred = np.reshape(a=np.random.uniform(size=13 * n), newshape=(n, 13))

qt_loss(y_true, y_pred)
