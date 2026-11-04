"""Assignment - using numpy and making a PR.

The goals of this assignment are:
    * Use numpy in practice with two easy exercises.
    * Use automated tools to validate the code (`pytest` and `ruff`)
    * Submit a Pull-Request on github to practice `git`.

The two functions below are skeleton functions. The docstrings explain what
are the inputs, the outputs and the expected error. Fill the function to
complete the assignment. The code should be able to pass the test that we
wrote. To run the tests, use `pytest test_numpy_question.py` at the root of
the repo. It should say that 2 tests ran with success.

We also ask to respect the pep8 convention: https://pep8.org.
This will be enforced with `ruff`. You can check that there are no errors
by calling `ruff check` at the root of the repo.
"""
import numpy as np



def max_index(X):
    """Return the row and column indices of the maximum value in a 2D array."""
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if X.ndim != 2:
        raise ValueError("Input must be 2D")

    i, j = np.unravel_index(np.argmax(X), X.shape)

    return int(i), int(j)


def wallis_product(n_terms):
    """Approximate pi using the Wallis product with n_terms."""
    product = 1.0

    for k in range(1, n_terms + 1):
        product *= (4 * k * k) / (4 * k * k - 1)

    return 2 * product

