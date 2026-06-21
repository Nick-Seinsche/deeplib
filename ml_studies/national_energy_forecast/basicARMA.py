"""
basicARMA module.

================

A minimalist implementation of an **ARMA(p, q)** process for univariate time
series.  The parameters are estimated via *exact* Gaussian maximum–likelihood
using the BFGS optimiser from *SciPy*.

Only the public API and formatting have been revised; *no executable logic has
been altered*.  The code now follows the most common **PEP 8** conventions and
includes extensive Google-style doc-strings.

Example
-------
```python
arma = basicARMA(p=2, q=1)
arma.fit(price_series)
forecast = arma.predict(price_series, horizon=12)
"""

import pandas as pd
import numpy as np
import math
import scipy.optimize


class basicARMA:
    """
    Gaussian ARMA(p, q) model.

    Parameters
    ----------
    p : int
        Order of the autoregressive (AR) part.  Must be non-negative.
    q : int
        Order of the moving-average (MA) part.  Must be non-negative.

    Attributes
    ----------
    p : int
        Stored AR order.
    q : int
        Stored MA order.
    mu : float | None
        Estimated unconditional mean of the series (initially *None*).
    phi : np.ndarray | None
        AR coefficients of length *p* (initially *None*).
    theta : np.ndarray | None
        MA coefficients of length *q* (initially *None*).
    log_sigma : float | None
        Log of the innovation variance (initially *None*).
    """

    def __init__(self, p: int, q: int) -> None:
        """Initialize the Class."""
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative integers")

        self.p = p
        self.q = q
        self.mu = None
        self.phi = None
        self.theta = None
        self.log_sigma = None

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _loss(Sigma: np.ndarray, X: pd.Series, p: int, q: int):
        """
        Negative log-likelihood of the ARMA model.

        The parameter vector has the form::

            Sigma = [phi_1, …, phi_p, theta_1, …, theta_q, μ, log(σ²)]

        The innovation variance is parameterised on the log-scale to ensure
        positivity.

        Args:
            Sigma: Flattened parameter vector of length *p + q + 2*.
            X: Observed time-series as a pandas ``Series``.
            p: AR order.
            q: MA order.

        Returns:
            The (unnormalised) negative log-likelihood.

        Raises:
            ValueError: If the shape or content of *Sigma* is invalid.
        """
        if len(Sigma) != p + q + 2:
            raise ValueError(f"Sigma must have length {p + q + 2}, got {len(Sigma)}")
        if not np.isfinite(Sigma).all():
            raise ValueError("Sigma must contain finite values")
        T = len(X)

        phi = Sigma[:p]
        theta = Sigma[p : p + q]
        mu = Sigma[-2]
        sigma = np.exp(Sigma[-1])

        eps = np.zeros(T)

        loss = T / 2 * np.log(2 * math.pi * sigma)

        start = max(p, q)

        for t in range(start, T):
            X_vec = [0] * max(p - t, 0) + list(X.values[max(t - p, 0) : t])
            eps_vec = [0] * max(q - t, 0) + list(eps[max(t - q, 0) : t])

            eps[t] = X[t] - mu - np.dot(phi, X_vec[::-1]) - np.dot(theta, eps_vec[::-1])
            loss += eps[t] ** 2 / (2 * sigma)

        return loss

    # def _fd_gradient(Sigma: np.ndarray, X: pd.Series, p: int, q: int):
    #     if len(Sigma) != p + q + 1:
    #         raise ValueError(f"Sigma must have length {p + q + 1}, got {len(Sigma)}")
    #     if not np.isfinite(Sigma).all():
    #         raise ValueError("Sigma must contain finite values")

    #     T = len(X)

    #     grad = np.zeros_like(Sigma)

    #     for i in range(len(grad)):
    #         grad[i] = (ARMA._loss(Sigma + np.eye(len(Sigma))[:, i] * 1e-5, X, p, q)
    #                    - ARMA._loss(Sigma - np.eye(len(Sigma))[:, i] * 1e-5, X, p, q))
    #         grad[i] /= 2 * 1e-5

    #     return grad

    def _bfgs_wrapper(X: pd.Series, Sigma: np.ndarray, p: int, q: int):
        """
        Run a BFGS optimisation step.

        Args:
            X: Observed time-series.
            Sigma: Initial parameter vector.
            p: AR order.
            q: MA order.

        Returns:
            np.ndarray: Optimised parameter vector.
        """
        result = scipy.optimize.minimize(
            fun=basicARMA._loss,
            x0=Sigma,
            args=(X, p, q),
            method="BFGS",
            # jac=ARMA._fd_gradient,
            options={"disp": True, "gtol": 1e-9, "maxiter": 5000},
        )

        return result.x

    # --------------------------------------------------------------------- #
    # Public interface                                                      #
    # --------------------------------------------------------------------- #

    def fit(self, X: pd.Series):
        """
        Estimate ARMA parameters by maximum likelihood.

        Args:
            X: A univariate pandas ``Series``.  Its index should be equally
                spaced because the forecasting logic relies on a fixed
                15-minute frequency when generating future timestamps.

        Raises:
            ValueError: If *X* is not a valid series for estimation.
        """
        T = len(X)

        if not isinstance(X, pd.Series):
            raise ValueError("X must be a pandas Series")
        if T < max(self.p, self.q):
            raise ValueError(
                f"X must have at least {max(self.p, self.q)} observations, got {T}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values")

        Sigma = basicARMA._bfgs_wrapper(
            X,
            np.eye(self.p + self.q + 2)[:, self.p + self.q + 1] * np.log(np.var(X)),
            self.p,
            self.q,
        )
        self.phi = Sigma[: self.p]
        self.theta = Sigma[self.p : self.p + self.q]
        self.mu = Sigma[-2]
        self.log_sigma = Sigma[-1]
        print("Fitted ARMA model with parameters:")
        print("phi:", self.phi)
        print("theta:", self.theta)
        print("mu=:", self.mu)
        print("log_sigma:", self.log_sigma)

    def predict(self, X: pd.Series, horizon: int = 1, mean: bool = True):
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        X : pd.Series
            Series already used in :pymeth:`fit`.  Only the tail is needed, but
            the full series is accepted for simplicity.
        horizon : int, default = 1
            Number of steps ahead to forecast.
        mean : bool, default = True
            If *True*, produce the conditional mean forecast.  If *False*,
            generate one random path by adding white noise with variance
            :math:`sigma^{2}`.

        Returns
        -------
        pd.Series
            Forecasted values indexed at a fixed **15-minute** frequency
            starting immediately after the last observation.
        """
        eps = np.zeros(len(X) + horizon)
        predictions = np.zeros(horizon)

        start = max(self.p, self.q)

        for t in range(start, len(X)):
            X_vec = [0] * max(self.p - t, 0) + list(X.values[max(t - self.p, 0) : t])
            eps_vec = [0] * max(self.q - t, 0) + list(eps[max(t - self.q, 0) : t])

            eps[t] = (
                X[t] - np.dot(self.phi, X_vec[::-1]) - np.dot(self.theta, eps_vec[::-1])
            )

        for t in range(horizon):
            X_vec = list(X.values[len(X) - self.p + t :]) + list(
                predictions[max(t - self.p, 0) : t]
            )
            eps_vec = eps[len(X) - self.q + t : len(X) + t]

            deterministic_part = (
                self.mu
                + np.dot(self.phi, X_vec[::-1])
                + np.dot(self.theta, eps_vec[::-1])
            )

            if mean:
                predictions[t] = deterministic_part
            else:
                predictions[t] = deterministic_part + np.random.normal(
                    0, np.exp(self.log_sigma) ** 0.5
                )

            eps[len(X) + t] = predictions[t] - deterministic_part

        return pd.Series(
            predictions,
            index=pd.date_range(
                start=X.index[-1] + pd.Timedelta(minutes=15),
                periods=horizon,
                freq="15T",
            ),
        )
