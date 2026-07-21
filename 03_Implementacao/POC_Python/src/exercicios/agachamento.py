"""
Implementacao do exercicio de agachamento.
"""

from exercicios.base_exercicio import ExercicioBase
from analisador.angulos import calcular_angulo
import cv2

class Agachamento(ExercicioBase):
    """
    Deteta agachamentos com base no angulo do joelho.
    """
    
    def __init__(self, nome_display="Agachamento", cor=(0,255,0), icone="AGACH"):
        super().__init__(nome_display, cor, icone)
        
        self.ANGULO_BAIXO = 70
        self.ANGULO_CIMA = 160
        self.em_baixo = False
        self.angulo_atual = 0
        
    def processar_landmarks(self, landmarks):
        if not landmarks:
            self.feedback = "Sem deteccao"
            return self.repeticoes, self.estado, self.feedback
            
        try:
            quadril = (
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].y
            )
            joelho = (
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].y
            )
            tornozelo = (
                landmarks.landmark[self.LANDMARKS.RIGHT_ANKLE.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_ANKLE.value].y
            )
            
            self.angulo_atual = calcular_angulo(quadril, joelho, tornozelo)
            
            if self.angulo_atual < self.ANGULO_BAIXO and not self.em_baixo:
                self.em_baixo = True
                self.estado = "em_baixo"
                self.feedback = "Sobe!"
            elif self.angulo_atual > self.ANGULO_CIMA and self.em_baixo:
                self.repeticoes += 1
                self.em_baixo = False
                self.estado = "em_cima"
                self.feedback = f"Rep {self.repeticoes} completa!"
                
        except Exception as e:
            self.feedback = f"Erro: {str(e)}"
            
        return self.repeticoes, self.estado, self.feedback
    
    def desenhar_info(self, imagem, landmarks):
        """Desenha informacoes com posicoes ajustadas para nao sair do ecra."""
        h, w, _ = imagem.shape
        
        # Nome no canto superior direito (ajustado)
        cv2.putText(imagem, f"{self.icone} {self.nome_display}", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cor, 2)
        
        # Angulo no joelho
        if hasattr(self, 'angulo_atual') and landmarks:
            try:
                joelho = landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value]
                pos = (int(joelho.x * w), int(joelho.y * h))
                # Garantir que nao desenha fora do ecra
                if pos[0] - 30 > 0 and pos[1] - 20 > 0:
                    cv2.putText(imagem, f"{self.angulo_atual:.0f}°", 
                               (pos[0]-30, pos[1]-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cor, 2)
            except:
                pass
    
    @staticmethod
    def testar():
        """Testa o exercicio com webcam."""
        print("Testando Agachamento... Pressiona 'q' para sair")
        import cv2
        from analisador.pose_detector import PoseDetector
        
        cap = cv2.VideoCapture(0)
        detector = PoseDetector()
        exercicio = Agachamento()
        
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
                
                cv2.putText(frame, f"Reps: {reps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, exercicio.cor, 2)
                cv2.putText(frame, feedback, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow("Teste Agachamento", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Agachamento.testar()