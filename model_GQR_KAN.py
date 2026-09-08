# ==========================================================
# GQR-KAN with one shared RBF-QR basis
# Kernel: exp(-r^2 / eps^2)
# Centers: [0, 1]
# Training/forward precision: selected in main.py
# One-time QR preprocessing: float64
# ==========================================================

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from train_utils import train


# ==========================================================
# Helper wrapper
# ==========================================================

class GQRFixedEPS:

    def __init__(
        self,
        a,
        *,
        num_grid,
        eps,
        alpha=1.0,
        m_total=None,
        max_extra=20,
        device="cpu",
        qr_dtype=torch.float64,
    ):
        self.model = GQRKAN(
            a=a,
            num_grid=num_grid,
            eps=eps,
            alpha=alpha,
            m_total=m_total,
            max_extra=max_extra,
            device=device,
            qr_dtype=qr_dtype,
        )

        self.train_hist = None
        self.eval_hist = None
        self.final_train_mse = None
        self.final_eval_rmse = None
        self.final_eval_rel = None
        self.final_eval_mae = None
        self.total_time = None

    def fit(
        self,
        x_train,
        y_train,
        x_eval,
        y_eval,
        epochs,
        lr,
        **unused_metadata,
    ):
        (
            self.train_hist,
            self.eval_hist,
            self.final_train_mse,
            self.final_eval_rmse,
            self.final_eval_rel,
            self.final_eval_mae,
            self.total_time,
        ) = train(
            self.model,
            x_train,
            y_train,
            x_eval,
            y_eval,
            epochs=epochs,
            lr=lr,
        )


# ==========================================================
# One fixed shared GQR basis
# ==========================================================

class GQRBasis1D(nn.Module):
    """
    Fixed one-dimensional RBF-QR basis.

    QR preprocessing is performed only once.
    The same basis is shared by every KAN layer.

    N = num_grid
    M = total number of Mercer eigenfunctions, M >= N
    """

    def __init__(
        self,
        *,
        num_grid: int,
        eps: float,
        alpha: float = 1.0,
        m_total: Optional[int] = None,
        max_extra: int = 20,
        device: str = "cpu",
        qr_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        if eps <= 0:
            raise ValueError(
                "eps must be positive because the kernel is "
                "exp(-r^2 / eps^2)."
            )

        if alpha <= 0:
            raise ValueError(
                "alpha must be positive."
            )

        if num_grid < 2:
            raise ValueError(
                "num_grid must be at least 2."
            )

        if max_extra < 0:
            raise ValueError(
                "max_extra must be nonnegative."
            )

        self.num_grid = int(num_grid)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.max_extra = int(max_extra)

        # Paper convention:
        #
        #     exp(-(ep_shape*r)^2)
        #
        # KAN convention:
        #
        #     exp(-r^2 / eps^2)
        #
        # Therefore:
        self.ep_shape = 1.0 / self.eps

        # --------------------------------------------------
        # QR centers
        # --------------------------------------------------
        #
        # Generate centers directly in qr_dtype.
        #
        # Creating centers in float32 and subsequently
        # converting them to float64 cannot restore the
        # precision already lost in their locations.
        #
        # This matters because the QR eigenfunction matrix
        # can be extremely ill-conditioned.
        # --------------------------------------------------

        centers_qr = torch.linspace(
            0.0,
            1.0,
            self.num_grid,
            device=device,
            dtype=qr_dtype,
        )

        # Runtime copy in the selected training dtype.
        centers = centers_qr.to(
            dtype=torch.get_default_dtype(),
            device=device,
        )

        self.register_buffer(
            "centers",
            centers,
        )

        # --------------------------------------------------
        # Select M
        # --------------------------------------------------

        self.m_total = self._select_m_total(
            num_grid=self.num_grid,
            ep_shape=self.ep_shape,
            alpha=self.alpha,
            m_total=m_total,
            max_extra=self.max_extra,
            dtype=qr_dtype,
        )

        # --------------------------------------------------
        # Construct QR correction
        # --------------------------------------------------

        CbarT = self._build_cbar_t(
            centers=centers_qr,
            ep_shape=self.ep_shape,
            alpha=self.alpha,
            m_total=self.m_total,
            qr_dtype=qr_dtype,
        )

        # Store the Mercer eigenvalue ratio for diagnostics.
        _, _, _, lambda_ratio = gqr_parameters_1d(
            ep_shape=self.ep_shape,
            alpha=self.alpha,
            dtype=qr_dtype,
            device=device,
        )

        self.register_buffer(
            "lambda_ratio",
            lambda_ratio.to(
                dtype=torch.get_default_dtype(),
                device=device,
            ),
        )

        # Forward uses the selected training dtype.
        self.register_buffer(
            "CbarT",
            CbarT.to(
                dtype=torch.get_default_dtype(),
                device=device,
            ),
        )

    def forward(self, x):
        """
        Evaluate the shared GQR basis.

        Input:
            x: [batch, input_dim]

        Output:
            psi: [batch, input_dim, num_grid]
        """

        phi = gaussian_mercer_phi_1d_recurrence(
            x=x,
            m_total=self.m_total,
            ep_shape=self.ep_shape,
            alpha=self.alpha,
        )

        N = self.num_grid

        # First N Mercer eigenfunctions.
        phi1 = phi[..., :N]

        if self.m_total == N:
            return phi1

        # Remaining M-N Mercer eigenfunctions.
        phi2 = phi[..., N:]

        CbarT = self.CbarT.to(
            dtype=x.dtype,
            device=x.device,
        )

        # Complete Section 4 RBF-QR correction:
        #
        #     psi = phi1 + phi2 @ CbarT
        correction = torch.matmul(
            phi2,
            CbarT,
        )

        return phi1 + correction

    # ======================================================
    # Select M
    # ======================================================

    @staticmethod
    def _select_m_total(
        *,
        num_grid: int,
        ep_shape: float,
        alpha: float,
        m_total: Optional[int],
        max_extra: int,
        dtype: torch.dtype,
    ) -> int:

        N = int(num_grid)

        # Manual M overrides automatic selection.
        if m_total is not None:
            M = int(m_total)

            if M < N:
                raise ValueError(
                    "Full QR basis requires "
                    "m_total >= num_grid. "
                    f"Got M={M}, N={N}."
                )

            return M

        # Automatic paper criterion:
        #
        #     lambda^(M-N) < machine epsilon
        _, _, _, lam = gqr_parameters_1d(
            ep_shape=ep_shape,
            alpha=alpha,
            dtype=dtype,
            device=torch.device("cpu"),
        )

        lam_float = float(lam)
        machine_eps = torch.finfo(dtype).eps

        if lam_float <= 0.0:
            auto_M = N

        elif lam_float >= 1.0:
            auto_M = N + max_extra

        else:
            auto_M = math.ceil(
                N
                + math.log(machine_eps)
                / math.log(lam_float)
            )

        capped_M = min(
            auto_M,
            N + int(max_extra),
        )

        return max(
            N,
            capped_M,
        )

    # ======================================================
    # Construct CbarT
    # ======================================================

    @staticmethod
    @torch.no_grad()
    def _build_cbar_t(
        *,
        centers,
        ep_shape: float,
        alpha: float,
        m_total: int,
        qr_dtype: torch.dtype,
    ):
        """
        Construct the Section 4 RBF-QR correction:

            Phi_Z = Q [R1 R2]

            Rhat = R1^{-1} R2

            D[m,n] = lambda^(m_index - n_index)

            CbarT = D .* Rhat^T
        """

        device = centers.device

        N = centers.numel()
        M = int(m_total)

        if M == N:
            return torch.empty(
                0,
                N,
                device=device,
                dtype=qr_dtype,
            )

        centers_qr = (
            centers
            .detach()
            .to(
                dtype=qr_dtype,
                device=device,
            )
            .view(-1, 1)
        )

        # Mercer functions at the fixed center points.
        phi_matrix = gaussian_mercer_phi_1d_recurrence(
            x=centers_qr,
            m_total=M,
            ep_shape=ep_shape,
            alpha=alpha,
        ).squeeze(1)

        # phi_matrix:
        #     [N, M]
        #
        # Q:
        #     [N, N]
        #
        # R:
        #     [N, M]
        _, R = torch.linalg.qr(
            phi_matrix,
            mode="reduced",
        )

        R1 = R[:, :N]
        R2 = R[:, N:]

        # --------------------------------------------------
        # MATLAB-compatible diagonal row scaling
        # --------------------------------------------------
        #
        # The supplied gqr_solveprep.m does:
        #
        #     iRdiag = diag(1 ./ diag(R1))
        #     R1s = iRdiag * R1
        #     Rhat = R1s \ (iRdiag * R2)
        #
        # This is algebraically equivalent to solving
        #
        #     R1 Rhat = R2
        #
        # but is numerically safer.
        # --------------------------------------------------

        diagonal = torch.diagonal(R1)

        if (
            not torch.isfinite(diagonal).all()
            or torch.any(diagonal == 0)
        ):
            raise RuntimeError(
                "RBF-QR preprocessing failed because R1 "
                "contains a zero or non-finite diagonal entry. "
                "Try another alpha or use float64."
            )

        inverse_diagonal = diagonal.reciprocal()

        R1_scaled = (
            inverse_diagonal[:, None]
            * R1
        )

        R2_scaled = (
            inverse_diagonal[:, None]
            * R2
        )

        Rhat = torch.linalg.solve_triangular(
            R1_scaled,
            R2_scaled,
            upper=True,
            left=True,
        )

        # Mercer eigenvalue ratio.
        _, _, _, lam = gqr_parameters_1d(
            ep_shape=ep_shape,
            alpha=alpha,
            dtype=qr_dtype,
            device=device,
        )

        # MATLAB indices are 1, 2, ..., M.
        idx1 = torch.arange(
            1,
            N + 1,
            device=device,
            dtype=qr_dtype,
        )

        idx2 = torch.arange(
            N + 1,
            M + 1,
            device=device,
            dtype=qr_dtype,
        )

        # Eigenvalue scaling:
        #
        #     D[m,n] = lambda^(m_index - n_index)
        D = lam ** (
            idx2[:, None]
            - idx1[None, :]
        )

        CbarT = D * Rhat.T

        if not torch.isfinite(CbarT).all():
            raise RuntimeError(
                "RBF-QR preprocessing produced a "
                "non-finite CbarT matrix. Try another "
                "alpha or use float64."
            )

        return CbarT


# ==========================================================
# One trainable KAN layer
# ==========================================================

class GQRKANLayer(nn.Module):
    """
    One KAN layer containing only trainable coefficients.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grid: int,
        device: str = "cpu",
    ):
        super().__init__()

        self.inputdim = int(input_dim)
        self.outdim = int(output_dim)
        self.num_grid = int(num_grid)

        self.coeffs = nn.Parameter(
            torch.empty(
                self.inputdim,
                self.outdim,
                self.num_grid,
                device=device,
            )
        )

        nn.init.normal_(
            self.coeffs,
            mean=0.0,
            std=1.0 / (
                self.inputdim
                * self.num_grid
            ),
        )

    def forward(self, psi):
        """
        Input:
            psi: [batch, input_dim, num_grid]

        Output:
            y: [batch, output_dim]
        """

        if psi.dim() != 3:
            raise ValueError(
                "Expected psi with shape "
                "[batch, input_dim, num_grid]."
            )

        if psi.shape[1] != self.inputdim:
            raise ValueError(
                f"Expected input dimension {self.inputdim}, "
                f"got {psi.shape[1]}."
            )

        if psi.shape[2] != self.num_grid:
            raise ValueError(
                f"Expected num_grid={self.num_grid}, "
                f"got {psi.shape[2]}."
            )

        return torch.einsum(
            "bin,ion->bo",
            psi,
            self.coeffs,
        )


# ==========================================================
# Multi-layer GQR-KAN
# ==========================================================

class GQRKAN(nn.Module):
    """
    Multi-layer GQR-KAN with one shared fixed RBF-QR basis.

    The QR preprocessing is performed once. At every layer,
    the shared basis is reevaluated at the current activations.
    """

    def __init__(
        self,
        a,
        *,
        num_grid: int,
        eps: float,
        alpha: float = 1.0,
        m_total: Optional[int] = None,
        max_extra: int = 20,
        device: str = "cpu",
        qr_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        if len(a) < 2:
            raise ValueError(
                "Architecture must contain at least "
                "an input and output dimension."
            )

        self.basis = GQRBasis1D(
            num_grid=num_grid,
            eps=eps,
            alpha=alpha,
            m_total=m_total,
            max_extra=max_extra,
            device=device,
            qr_dtype=qr_dtype,
        )

        self.layers = nn.ModuleList([
            GQRKANLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                num_grid=num_grid,
                device=device,
            )
            for input_dim, output_dim
            in zip(a[:-1], a[1:])
        ])

    def forward(self, x):

        for layer in self.layers:
            # Evaluate the same stable basis at the
            # current layer activations.
            psi = self.basis(x)

            # Apply trainable KAN coefficients.
            x = layer(psi)

        return x


# ==========================================================
# Gaussian Mercer parameters
# ==========================================================

def gqr_parameters_1d(
    *,
    ep_shape: float,
    alpha: float,
    dtype: torch.dtype,
    device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Return:

        ep
        beta
        delta^2
        lambda

    using equations (3.4) and (3.5) from the paper.
    """

    ep = torch.tensor(
        float(ep_shape),
        dtype=dtype,
        device=device,
    )

    a = torch.tensor(
        float(alpha),
        dtype=dtype,
        device=device,
    )

    beta = (
        1.0
        + (2.0 * ep / a) ** 2
    ) ** 0.25

    # Match the supplied MATLAB implementation.
    # Use an asymptotic expansion near beta=1 to avoid
    # catastrophic cancellation.
    if float(beta - 1.0) < 1.0e-4:
        delta2 = (
            ep**2
            - ep**4 / a**2
            + 2.0 * ep**6 / a**4
        )
    else:
        delta2 = (
            0.5
            * a**2
            * (beta**2 - 1.0)
        )

    lam = (
        ep**2
        / (
            a**2
            + ep**2
            + delta2
        )
    )

    return (
        ep,
        beta,
        delta2,
        lam,
    )


# ==========================================================
# Gaussian Mercer eigenfunctions
# ==========================================================

def gaussian_mercer_phi_1d_recurrence(
    *,
    x,
    m_total: int,
    ep_shape: float,
    alpha: float,
):
    """
    Evaluate the first M normalized Gaussian Mercer
    eigenfunctions with the Hermite recurrence.

    Input:
        x: [batch, input_dim]

    Output:
        phi: [batch, input_dim, M]
    """

    if m_total < 1:
        raise ValueError(
            "m_total must be at least 1."
        )

    dtype = x.dtype
    device = x.device

    _, beta, delta2, _ = gqr_parameters_1d(
        ep_shape=ep_shape,
        alpha=alpha,
        dtype=dtype,
        device=device,
    )

    p_list = []

    # First normalized Mercer eigenfunction.
    p1 = (
        torch.sqrt(beta)
        * torch.exp(
            -delta2 * x**2
        )
    )

    p_list.append(p1)

    # Second normalized Mercer eigenfunction.
    if m_total >= 2:
        p2 = (
            math.sqrt(2.0)
            * beta
            * alpha
            * x
            * p1
        )

        p_list.append(p2)

    # Remaining normalized Mercer eigenfunctions:
    #
    # phi_(k+1) =
    #     sqrt(2/k) beta alpha x phi_k
    #     - sqrt((k-1)/k) phi_(k-1)
    for k in range(2, m_total):

        c1 = (
            math.sqrt(2.0 / k)
            * beta
            * alpha
        )

        c2 = math.sqrt(
            (k - 1.0) / k
        )

        p_next = (
            c1
            * x
            * p_list[-1]
            - c2
            * p_list[-2]
        )

        p_list.append(p_next)

    return torch.stack(
        p_list,
        dim=-1,
    )