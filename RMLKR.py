import time
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize
import numpy as np
from sklearn.base import TransformerMixin
from metric_learn.base_metric import MahalanobisMixin
from metric_learn import MLKR
from metric_learn._util import _initialize_components, _check_n_components

class RMLKR(MLKR, MahalanobisMixin, TransformerMixin):
    """ Regularized metric learning for kernel regression.
    This is entirely based on the MLKR class available in metric_learn package. The only difference
    is the inclusion of a regularization term in the loss function and the derivatives.

    Parameters
  ----------
  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  init : string or numpy array, optional (default='auto')
    Initialization of the linear transformation. Possible options are
    'auto', 'pca', 'identity', 'random', and a numpy array of shape
    (n_features_a, n_features_b).

    'auto'
      Depending on ``n_components``, the most reasonable initialization
      will be chosen. If ``n_components < min(n_features, n_samples)``,
      we use 'pca', as it projects data in meaningful directions (those
      of higher variance). Otherwise, we just use 'identity'.

    'pca'
      ``n_components`` principal components of the inputs passed
      to :meth:`fit` will be used to initialize the transformation.
      (See `sklearn.decomposition.PCA`)

    'identity'
      If ``n_components`` is strictly smaller than the
      dimensionality of the inputs passed to :meth:`fit`, the identity
      matrix will be truncated to the first ``n_components`` rows.

    'random'
      The initial transformation will be a random array of shape
      `(n_components, n_features)`. Each value is sampled from the
      standard normal distribution.

    numpy array
      n_features_b must match the dimensionality of the inputs passed to
      :meth:`fit` and n_features_a must be less than or equal to that.
      If ``n_components`` is not None, n_features_a must match it.

  tol : float, optional (default=None)
    Convergence tolerance for the optimization.

  max_iter : int, optional (default=1000)
    Cap on number of conjugate gradient iterations.

  verbose : bool, optional (default=False)
    Whether to print progress messages or not.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to initialize the random
    transformation. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the transformation.

  lambda_ : regularization hyperparameters.

  Attributes
  ----------
  n_iter_ : `int`
    The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.
    """

    def __init__(self, n_components=None, init='auto', tol=None, max_iter=1000, verbose=False, preprocessor=None, random_state=None, lambda_= 0.5):
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.lambda_ = lambda_
        super(MLKR, self).__init__(preprocessor)

    def _regularized_loss(self, flatA, X, y, lambda_):
        """
        Compute the regularized loss function.
        """
        
        cost, grad = self._loss(flatA, X, y)

        cost_lambda = 0.5 * lambda_ * np.sum(flatA ** 2)
        grad_lambda = lambda_ * flatA

        return cost+ cost_lambda, grad + grad_lambda

    def fit(self, X, y):
      """
      Fit MLKR model

      Parameters
      ----------
      X : (n x d) array of samples
      y : (n) data labels
      """
      X, y = self._prepare_inputs(X, y, y_numeric=True,
                                  ensure_min_samples=2)
      n, d = X.shape
      if y.shape[0] != n:
          raise ValueError('Data and label lengths mismatch: %d != %d'
                           % (n, y.shape[0]))

      m = _check_n_components(d, self.n_components)
      m = self.n_components
      if m is None:
          m = d
      # if the init is the default (None), we raise a warning
      A = _initialize_components(m, X, y, init=self.init,
                                 random_state=self.random_state,
                                 # MLKR works on regression targets:
                                 has_classes=False)

      # Measure the total training time
      train_time = time.time()

      self.n_iter_ = 0
      res = minimize(self._regularized_loss, A.ravel(), (X, y, self.lambda_), method='L-BFGS-B',
                     jac=True, tol=self.tol,
                     options=dict(maxiter=self.max_iter))
      self.components_ = res.x.reshape(A.shape)

      # Stop timer
      train_time = time.time() - train_time
      if self.verbose:
          cls_name = self.__class__.__name__
          # Warn the user if the algorithm did not converge
          if not res.success:
              warnings.warn('[{}] MLKR did not converge: {}'
                            .format(cls_name, res.message), ConvergenceWarning)
          print('[{}] Training took {:8.2f}s.'.format(cls_name, train_time))

      return self