"""N4SID: Subspace State-Space System Identification.

Fits a discrete-time linear state-space model:
    x_{k+1} = A x_k + B u_k
    y_k     = C x_k + D u_k

where x is the latent state, u is the input, y is the observation.
If n_states == n_outputs, C ~ I and we work directly in observable space.

Reference: Van Overschee & De Moor, "Subspace Identification for Linear
Systems" (1996).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _block_hankel(data: NDArray, n_block_rows: int) -> NDArray:
    """Build a block Hankel matrix from data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    n_block_rows : int
        Number of block rows (determines subspace dimension)

    Returns
    -------
    H : ndarray, shape (n_channels * n_block_rows, n_cols)
    """
    n_ch, n_samp = data.shape
    n_cols = n_samp - n_block_rows + 1
    H = np.zeros((n_ch * n_block_rows, n_cols))
    for i in range(n_block_rows):
        H[i * n_ch : (i + 1) * n_ch, :] = data[:, i : i + n_cols]
    return H


class N4SIDModel:
    """Linear state-space model identified via N4SID.

    Parameters
    ----------
    n_states : int
        Latent state dimension. If equal to n_outputs, the model
        works directly in observable space (C ~ I).
    n_block_rows : int
        Number of block rows in Hankel matrices. Must be > n_states.
        Larger values use more data but are more robust.
    """

    def __init__(self, n_states: int, n_block_rows: int = 20):
        self.n_states = n_states
        self.n_block_rows = n_block_rows

        # System matrices (set after fit)
        self.A: NDArray | None = None  # (n_states, n_states)
        self.B: NDArray | None = None  # (n_states, n_inputs)
        self.C: NDArray | None = None  # (n_outputs, n_states)
        self.D: NDArray | None = None  # (n_outputs, n_inputs)

        # Fit metadata
        self.singular_values: NDArray | None = None
        self.n_outputs: int | None = None
        self.n_inputs: int | None = None

    def fit(self, y: NDArray, u: NDArray) -> N4SIDModel:
        """Fit state-space model from input-output data.

        Parameters
        ----------
        y : ndarray, shape (n_outputs, n_samples)
            Output (observation) trajectory
        u : ndarray, shape (n_inputs, n_samples)
            Input trajectory

        Returns
        -------
        self
        """
        n_y, N = y.shape
        n_u = u.shape[0]
        self.n_outputs = n_y
        self.n_inputs = n_u
        i = self.n_block_rows  # shorthand
        n = self.n_states

        assert u.shape[1] == N, "u and y must have same number of samples"
        assert i > n, "n_block_rows must be > n_states"

        # Build block Hankel matrices
        # Past and future for outputs
        Y = _block_hankel(y, 2 * i)
        U = _block_hankel(u, 2 * i)

        j = Y.shape[1]  # number of columns

        # Split into past and future
        Y_p = Y[: i * n_y, :]  # past outputs
        Y_f = Y[i * n_y :, :]  # future outputs
        U_p = U[: i * n_u, :]  # past inputs
        U_f = U[i * n_u :, :]  # future inputs

        # Construct the "past" data matrix
        W_p = np.vstack([U_p, Y_p])  # (i*(n_u+n_y), j)

        # Oblique projection: project Y_f onto W_p along U_f
        # O_i = Y_f / U_f^perp * W_p
        # Using QR or direct least squares
        top = np.vstack([U_f, W_p])  # ((i*n_u + i*(n_u+n_y)), j)
        # Solve: Y_f = [L1 L2] @ [U_f; W_p] via least squares using QR decomposition
        Q, R = np.linalg.qr(top.T, mode='reduced')
        L = np.linalg.solve(R, (Y_f @ Q).T).T
        L2 = L[:, i * n_u :]  # coefficients for W_p
        O_i = L2 @ W_p  # oblique projection (i*n_y, j)

        # SVD of the oblique projection
        U_svd, S, Vt = np.linalg.svd(O_i, full_matrices=False)
        self.singular_values = S

        # Truncate to n_states
        U1 = U_svd[:, :n]
        S1 = np.diag(S[:n])
        V1t = Vt[:n, :]

        # Observability matrix and state sequence
        Gamma = U1 @ np.sqrt(S1)  # (i*n_y, n)
        X = np.sqrt(S1) @ V1t  # (n, j) — state sequence

        # Extract C from the first block of Gamma
        self.C = Gamma[:n_y, :]

        # Extract A, B from state sequence
        # X_{k+1} = A X_k + B U_k
        # Y_k = C X_k + D U_k
        # Using the state sequence at consecutive times:
        X_curr = X[:, :-1]  # (n, j-1)
        X_next = X[:, 1:]  # (n, j-1)
        U_curr = u[:, i : i + X_curr.shape[1]]  # align with state sequence
        Y_curr = y[:, i : i + X_curr.shape[1]]

        # Solve [X_next; Y_curr] = [A B; C D] @ [X_curr; U_curr]
        lhs = np.vstack([X_next, Y_curr])  # (n + n_y, j-1)
        rhs = np.vstack([X_curr, U_curr])  # (n + n_u, j-1)
        # Solve: lhs = ABCD @ rhs via QR decomposition of rhs.T
        Q_rhs, R_rhs = np.linalg.qr(rhs.T, mode='reduced')
        ABCD = np.linalg.solve(R_rhs, (lhs @ Q_rhs).T).T

        self.A = ABCD[:n, :n]
        self.B = ABCD[:n, n:]
        # C and D from lower block (refine C)
        self.C = ABCD[n:, :n]
        self.D = ABCD[n:, n:]

        return self

    def predict(
        self,
        y0: NDArray,
        u: NDArray,
    ) -> NDArray:
        """Forward-simulate the identified system.

        Parameters
        ----------
        y0 : ndarray, shape (n_outputs,)
            Initial observation (used to estimate initial state)
        u : ndarray, shape (n_inputs, n_steps)
            Input trajectory

        Returns
        -------
        y_pred : ndarray, shape (n_outputs, n_steps)
            Predicted output trajectory
        """
        assert self.A is not None, "Model not fitted yet. Call fit() first."

        n_steps = u.shape[1]
        n_y = self.n_outputs
        n = self.n_states

        # Estimate initial state from initial observation
        # x0 = C^+ @ y0
        x = np.linalg.pinv(self.C) @ y0

        y_pred = np.zeros((n_y, n_steps))
        for k in range(n_steps):
            y_pred[:, k] = self.C @ x + self.D @ u[:, k]
            x = self.A @ x + self.B @ u[:, k]

        return y_pred

    def eigenvalues(self) -> NDArray:
        """Return eigenvalues of the state transition matrix A."""
        assert self.A is not None, "Model not fitted"
        return np.linalg.eigvals(self.A)

    def is_stable(self) -> bool:
        """Check if all eigenvalues of A are inside the unit circle."""
        return bool(np.all(np.abs(self.eigenvalues()) < 1.0))
