"""
CycleLayer: Physics-informed Brayton Cycle Layer for turbofan RUL prognostics.

Architecture overview:
    Sensor data → Encoder → θ (cycle parameters) → BraytonCycleLayer → features → Prognostics → RUL
"""

from cyclelayer.models.cycle_layer import CycleLayerNet

__version__ = "0.1.0"
__all__ = ["CycleLayerNet"]
