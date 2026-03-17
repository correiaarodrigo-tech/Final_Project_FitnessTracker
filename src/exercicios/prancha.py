"""
Implementacao do exercicio de prancha (plank).
"""

from exercicios.base_exercicio import ExercicioBase
from analisador.angulos import calcular_angulo
import cv2
import time

class Prancha(ExercicioBase):
    """
    Deteta prancha mantendo angulo do corpo proximo de 180°.
    """
    
    def __init__(self, nome_display="Prancha", cor=(255, 255, 0), icone="PRANCHA"):
        super().__init__(nome_display, cor, icone)
        
        # Angulos de referencia
        self.ANGULO_IDEAL = 170
        self.TOLERANCIA = 15
        
        self.tempo_inicio = None
        self.tempo_total = 0
        self.posicao_correta = False
        self.angulo_atual = 0
        
    def processar_landmarks(self, landmarks):
        """
        Avalia a posicao de prancha.
        """
        if not landmarks:
            self.feedback = "Sem deteccao"
            return self.repeticoes, self.estado, self.feedback
            
        try:
            # Extrair pontos
            ombro = (
                landmarks.landmark[self.LANDMARKS.RIGHT_SHOULDER.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_SHOULDER.value].y
            )
            quadril = (
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].y
            )
            joelho = (
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].y
            )
            
            # Calcular angulo do corpo
            self.angulo_atual = calcular_angulo(ombro, quadril, joelho)
            
            # Verificar se esta em posicao correta
            if abs(self.angulo_atual - self.ANGULO_IDEAL) <= self.TOLERANCIA:
                if not self.posicao_correta:
                    self.posicao_correta = True
                    self.tempo_inicio = time.time() if self.tempo_inicio is None else self.tempo_inicio
                    
                if self.tempo_inicio:
                    self.tempo_total = time.time() - self.tempo_inicio
                    
                self.estado = "posicao_correta"
                self.feedback = f"Mantem! {self.tempo_total:.1f}s"
            else:
                if self.posicao_correta:
                    self.posicao_correta = False
                    
                self.estado = "ajustar"
                self.feedback = "Ajusta a posicao - corpo reto"
                
            self.repeticoes = int(self.tempo_total / 10)
            
        except Exception as e:
            self.feedback = f"Erro: {str(e)}"
            
        return self.repeticoes, self.estado, self.feedback
    
    def desenhar_info(self, imagem, landmarks):
        """Desenha informacoes da prancha."""
        # Nome no canto superior direito - ajustado para nao sair do ecra
        h, w, _ = imagem.shape
        cv2.putText(imagem, f"{self.icone} {self.nome_display}", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cor, 2)
        
        # Mostrar tempo
        if self.tempo_total > 0:
            cv2.putText(imagem, f"Tempo: {self.tempo_total:.1f}s", 
                       (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cor, 2)
    
    @staticmethod
    def testar():
        """Testa prancha com webcam."""
        print("🧪 Testando Prancha... Pressiona 'q' para sair")
        import cv2
        from analisador.pose_detector import PoseDetector
        
        cap = cv2.VideoCapture(0)
        detector = PoseDetector()
        exercicio = Prancha()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks = detector.encontrar_landmarks(frame)
            detector.desenhar_landmarks(frame, landmarks)
            
            if landmarks:
                reps, estado, feedback = exercicio.processar_landmarks(landmarks)
                exercicio.desenhar_info(frame, landmarks)
                
                cv2.putText(frame, "PRANCHA", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, exercicio.cor, 2)
                cv2.putText(frame, feedback, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow("Teste Prancha", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Prancha.testar()