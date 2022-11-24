import numpy as np


from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

base = importr("base")
utils = importr("utils")
stats = importr("stats4")

utils.chooseCRANmirror(ind=1)
# utils.install_packages("MultiRNG")
print(np.__version__)

np_cv_rules = default_converter + numpy2ri.converter

scoringRules = importr("scoringRules")
with localconverter(np_cv_rules) as cv:
    y = np.array([0, 1, 2, 3, 4])
    print(scoringRules.flapl(x=0.5, location=0, scale=1))

multiRNG = importr("MultiRNG")
print(multiRNG.__version__)

sigma = np.ones(shape=(3, 3))
for i in range(3):
    for j in range(3):
        sigma[i, j] = 0.5 ** abs(i - j)

with localconverter(np_cv_rules) as cv:
    print(multiRNG.draw_d_variate_uniform(no_row=10, d=3, cov_mat=sigma))
