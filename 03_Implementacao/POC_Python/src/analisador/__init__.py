"""
Módulo de análise de pose e cálculos geométricos.
Disponibiliza os componentes principais para deteção e processamento.
"""

from .pose_detector import PoseDetector
from .angulos import calcular_angulo

__all__ = ['PoseDetector', 'calcular_angulo']