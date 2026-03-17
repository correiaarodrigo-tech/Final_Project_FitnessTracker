"""
Classe base abstrata para todos os exercicios.
Define o contrato e funcionalidades comuns.
"""

from abc import ABC, abstractmethod
import mediapipe as mp
import cv2

class ExercicioBase(ABC):
    """
    Classe abstrata que define o contrato para todos os exercicios.
    """
    
    LANDMARKS = mp.solutions.pose.PoseLandmark
    
    def __init__(self, nome_display="Exercicio", cor=(255,255,255), icone="EXE"):
        """
        Args:
            nome_display: Nome para mostrar no ecra
            cor: Cor BGR para elementos visuais
            icone: Texto curto (sem emojis)
        """
        self.nome_display = nome_display
        self.cor = cor
        self.icone = icone
        
        self.repeticoes = 0
        self.estado = "aguardando"
        self.feedback = ""
        
    @abstractmethod
    def processar_landmarks(self, landmarks):
        """
        Processa os landmarks e atualiza estado.
        
        Returns:
            tuple: (repeticoes, estado, feedback)
        """
        pass
    
    def desenhar_info(self, imagem, landmarks):
        """
        Desenha informacoes especificas.
        Versao base: mostra icone e nome.
        """
        h, w, _ = imagem.shape
        texto = f"{self.icone} {self.nome_display}"
        cv2.putText(imagem, texto, (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cor, 2)
    
    def reset(self):
        """Reinicia o exercicio."""
        self.repeticoes = 0
        self.estado = "aguardando"
        self.feedback = ""