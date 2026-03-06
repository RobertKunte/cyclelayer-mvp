"""Loss functions for RUL prognostics training.

RULLoss
    Weighted MSE + asymmetric over-estimation penalty.

PhysicsInformedLoss
    RULLoss + thermodynamic feasibility + efficiency monotonicity.

CompositeLoss  (Phase 2 multi-task)
    L_total = L_rul + lambda_theta * L_theta
    where L_theta supervises theta_hat against theta_true from the dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RULLoss(nn.Module):
    """Combined MSE + asymmetric penalty loss for RUL regression.

    The asymmetric term penalizes over-estimation of RUL more than
    under-estimation, matching the N-CMAPSS scoring convention.

    Args:
        mse_weight: Weight for the standard MSE term.
        asymmetry: If > 0, adds an asymmetric penalty α * max(ŷ - y, 0)^2.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        asymmetry: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.asymmetry = asymmetry
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute loss.

        Args:
            pred:   Predicted RUL, shape (B,).
            target: Ground-truth RUL, shape (B,).
        """
        mse = F.mse_loss(pred, target, reduction=self.reduction)

        if self.asymmetry > 0.0:
            over = F.relu(pred - target)   # > 0 only when over-estimating
            asym = (over ** 2).mean() if self.reduction == "mean" else (over ** 2).sum()
            return self.mse_weight * mse + self.asymmetry * asym

        return self.mse_weight * mse


class PhysicsInformedLoss(nn.Module):
    """Physics-aware composite loss.

    Combines the RUL regression loss with penalty terms derived from the
    thermodynamic cycle parameters θ produced by the encoder.

    Physics penalties
    -----------------
    feasibility_weight  (λ_phys):
        Penalizes negative net work output.  For a Brayton cycle,
        W_net = cp * (T3 - T4) - cp * (T2 - T1) must be positive.
        We approximate: penalty = ReLU(-W_net_norm) where W_net_norm is
        the first feature from BraytonCycleLayer.

    monotonicity_weight (λ_mono):
        For a mini-batch sampled as a *trajectory segment*, the thermal
        efficiency η_th (feature index 1) should not increase as the
        engine degrades.  Penalty = ReLU(Δη_th) summed over the batch.
        Only meaningful when the batch preserves temporal order.

    Args:
        rul_loss: Base :class:`RULLoss` instance.
        feasibility_weight: Weight for the W_net ≥ 0 constraint.
        monotonicity_weight: Weight for the η non-increasing constraint.
    """

    def __init__(
        self,
        rul_loss: RULLoss | None = None,
        feasibility_weight: float = 0.1,
        monotonicity_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.rul_loss = rul_loss or RULLoss()
        self.feasibility_weight = feasibility_weight
        self.monotonicity_weight = monotonicity_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        cycle_features: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute composite loss and return component breakdown.

        Args:
            pred:           Predicted RUL (B,).
            target:         True RUL (B,).
            cycle_features: Output of BraytonCycleLayer (B, n_features).
                            Feature layout (see brayton_cycle.py):
                              0: W_net_norm, 1: η_th, 2: r_bw, ...

        Returns:
            total: Scalar loss tensor.
            components: Dict with keys ``"rul"``, ``"feasibility"``,
                        ``"monotonicity"``.
        """
        rul = self.rul_loss(pred, target)

        # Physics penalty 1: W_net >= 0
        W_net_norm = cycle_features[:, 0]
        feasibility = F.relu(-W_net_norm).mean()

        # Physics penalty 2: η_th non-increasing (temporal monotonicity)
        eta_th = cycle_features[:, 1]
        if eta_th.shape[0] > 1:
            delta_eta = eta_th[1:] - eta_th[:-1]           # positive = increasing
            monotonicity = F.relu(delta_eta).mean()
        else:
            monotonicity = torch.zeros(1, device=pred.device)

        total = (
            rul
            + self.feasibility_weight * feasibility
            + self.monotonicity_weight * monotonicity
        )

        components = {
            "rul": rul.detach(),
            "feasibility": feasibility.detach(),
            "monotonicity": monotonicity.detach(),
        }
        return total, components


# ---------------------------------------------------------------------------
# CompositeLoss — multi-task RUL + theta supervision (Phase 2)
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """Multi-task loss for CycleLayerNetV1.

    L_total = L_rul + lambda_theta * L_theta

    L_rul   — RULLoss (MSE + asymmetry).
    L_theta — Huber loss between theta_hat and theta_true.
              Huber is more robust to the initial random theta_hat values
              than plain MSE.

    Args:
        rul_loss: Base RULLoss instance.
        lambda_theta: Weight for the health-parameter supervision term.
        huber_delta: Delta parameter for Huber (smooth-L1) loss on theta.
    """

    def __init__(
        self,
        rul_loss: RULLoss | None = None,
        lambda_theta: float = 0.1,
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__()
        self.rul_loss = rul_loss or RULLoss()
        self.lambda_theta = lambda_theta
        self.huber_delta = huber_delta

    def forward(
        self,
        rul_pred: Tensor,
        rul_true: Tensor,
        theta_hat: Tensor,
        theta_true: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute composite loss.

        Args:
            rul_pred:   Predicted RUL (B,).
            rul_true:   Ground-truth RUL (B,).
            theta_hat:  Predicted health params (B, n_health_params).
            theta_true: True health params (B, n_health_params).

        Returns:
            total: Scalar loss.
            components: Dict with keys "rul" and "theta".
        """
        L_rul   = self.rul_loss(rul_pred, rul_true)
        L_theta = F.huber_loss(theta_hat, theta_true, delta=self.huber_delta)
        total   = L_rul + self.lambda_theta * L_theta
        return total, {"rul": L_rul.detach(), "theta": L_theta.detach()}
