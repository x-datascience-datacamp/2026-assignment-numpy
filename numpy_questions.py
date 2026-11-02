"""NumPy assignment: find a maximum and approximate pi."""

import numpy as np


def max_index(X):
    """Return the row and column of the maximum in a 2D array."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("Input must be a two-dimensional NumPy array.")

    flat_index = np.argmax(X)
    i, j = np.unravel_index(flat_index, X.shape)
    return int(i), int(j)


def wallis_product(n_terms):
    """Approximate pi using the specified number of Wallis terms."""
    k = np.arange(1, n_terms + 1, dtype=float)
    factors = (4 * k**2) / (4 * k**2 - 1)
    return float(2 * np.prod(factors))
