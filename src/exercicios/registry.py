"""
Registry de exercicios usando ENUMs.
"""

from config.teclas import TeclaExercicio
from exercicios.agachamento import Agachamento
from exercicios.prancha import Prancha
from exercicios.flexao import Flexao
from exercicios.lunge import Lunge  

class ExercicioRegistry:
    """Mapeia ENUMs para classes de exercicios."""
    
    _EXERCICIOS = {
        TeclaExercicio.AGACHAMENTO: {
            'classe': Agachamento,
            'nome_display': 'Agachamento',
            'cor': (0, 255, 0),
            'icone': 'AGACH',  
            'dificuldade': 'iniciante',
            'descricao': 'Mantem costas retas e desce ate 90°'
        },
        TeclaExercicio.PRANCHA: {
            'classe': Prancha,
            'nome_display': 'Prancha',
            'cor': (255, 255, 0),
            'icone': 'PRANCHA',
            'dificuldade': 'intermedio',
            'descricao': 'Mantem corpo reto'
        },
        TeclaExercicio.FLEXAO: {
            'classe': Flexao,
            'nome_display': 'Flexao',
            'cor': (0, 255, 255),
            'icone': 'FLEX',
            'dificuldade': 'avancado',
            'descricao': 'Cotovelos a 90°'
        },
        TeclaExercicio.LUNGE: {
            'classe': Lunge,
            'nome_display': 'Lunge',
            'cor': (255, 0, 255),
            'icone': 'LUNGE',
            'dificuldade': 'intermedio',
            'descricao': 'Avanco alternado com joelho a 90°'
        }
    }
    
    @classmethod
    def get_exercicio(cls, tecla_enum):
        """Retorna instancia do exercicio."""
        if tecla_enum not in cls._EXERCICIOS:
            return None
            
        config = cls._EXERCICIOS[tecla_enum]
        return config['classe'](
            nome_display=config['nome_display'],
            cor=config['cor'],
            icone=config['icone']
        )
    
    @classmethod
    def listar_todos(cls):
        """Lista todos exercicios."""
        for tecla, config in cls._EXERCICIOS.items():
            print(f"  {tecla.value}: {config['nome_display']} ({config['dificuldade']})")