"""
Modulo de exercicios fisicos.
"""

from .base_exercicio import ExercicioBase
from .agachamento import Agachamento
from .prancha import Prancha
from .flexao import Flexao
from .registry import ExercicioRegistry

__all__ = [
    'ExercicioBase', 
    'Agachamento', 
    'Prancha', 
    'Flexao',
    'ExercicioRegistry'
]