"""
ENUM para todas as teclas de atalho do sistema.
"""

from enum import Enum

class TeclaExercicio(Enum):
    """Teclas para selecionar exercicios."""
    AGACHAMENTO = 'a'
    PRANCHA = 'p'
    FLEXAO = 'f'
    LUNGE = 'l'      
    
    @classmethod
    def listar_disponiveis(cls):
        """Retorna string com todas as teclas."""
        return ' '.join([f"{e.value}:{e.name}" for e in cls])
    
    @classmethod
    def from_tecla(cls, tecla):
        """Converte tecla no ENUM correspondente."""
        if tecla is None:
            return None
        for e in cls:
            if e.value == tecla:
                return e
        return None