# RMLKR
Regularized version of the MLKR algorithm implemented in metric_learning package

This module is practically the same as the original `MLKR` module implemented in `metric_learn` package with the slight difference that the leave-one-out loss has been modified from
$$\mathcal{L} = \sum_i (y - \hat{y})^2,$$ to $$\mathcal{L} = \sum_i (y - \hat{y})^2 + \lambda/2 ||A||_2 ^2,$$ to regularize the loss function for regression purposes.