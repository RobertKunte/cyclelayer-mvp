"""Loss functions package (V3.1b+).

Legacy losses (`RULLoss`, `PhysicsInformedLoss`, `CompositeLoss`) live in
`cyclelayer.training.losses` for backward compatibility. New V3 losses
live in this package.
"""

from cyclelayer.losses.cyclelayer_v3_losses import (
    CycleLayerV3Loss,
    V3LossConfig,
)

__all__ = ["CycleLayerV3Loss", "V3LossConfig"]
