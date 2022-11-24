import numpy as np
from random import choices
from fn_eval import bern_quants
from tqdm import tqdm
import cProfile
import pstats
import scipy.stats as ss
import multiprocessing.pool as mp
import functools
from time import time_ns


def fn_apply_bqn(i, n_ens, **kwargs):
    # Sample individual distribution
    i_rep = choices(
        population=range(n_ens), k=kwargs["n_lp_samples"]
    )  # with replacement

    alpha_vec = [
        bern_quants(
            alpha=kwargs["f_ls"][f"alpha{j}"][i, :],
            q_levels=np.random.uniform(size=1),
        )
        for j in i_rep
    ]
    # print(len(alpha_vec))
    # print(alpha_vec[0].shape)
    return np.reshape(np.asarray(alpha_vec), newshape=(kwargs["n_lp_samples"]))


def main():

    n_ens = 10
    y_test_size = 10000
    y_test = np.random.uniform(size=y_test_size)

    def this_local_function():
        print(n_ens)

    this_local_function()
    f_ls = {
        "alpha0": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha1": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha2": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha3": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha4": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha5": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha6": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha7": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha8": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
        "alpha9": np.reshape(
            np.random.uniform(size=100 * y_test_size),
            newshape=(y_test_size, 100),
        ),
    }

    kwargs = {"n_ens": 10, "n_lp_samples": 100}

    start_time = time_ns()
    pool = mp.Pool(1)
    # result_parallel = pool.map(
    #    func=functools.partial(
    #        fn_apply_bqn, **dict(kwargs, n_ens=n_ens, f_ls=f_ls)
    #    ),
    #    iterable=tqdm(range(len(y_test))),
    # )
    result = list(
        map(
            lambda x: fn_apply_bqn(x, **dict(kwargs, n_ens=n_ens, f_ls=f_ls)),
            tqdm(range(len(y_test))),
        )
    )

    result = np.asarray(result)
    end_time = time_ns()

    print(end_time - start_time)
    # print(result)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
