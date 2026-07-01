"""
Implementacao do exercicio de flexao de bracos (push-up).
"""

from exercicios.base_exercicio import ExercicioBase
from analisador.angulos import calcular_angulo
import cv2

class Flexao(ExercicioBase):
    """
    Deteta flexoes com base no angulo do cotovelo.
    """
    
    def __init__(self, nome_display="Flexao", cor=(0,255,255), icone="FLEX"):
        super().__init__(nome_display, cor, icone)
        
        self.ANGULO_BAIXO = 70
        self.ANGULO_CIMA = 150
        self.em_baixo = False
        self.angulo_atual = 0
        
    def processar_landmarks(self, landmarks):
        if not landmarks:
            self.feedback = "Sem deteccao"
            return self.repeticoes, self.estado, self.feedback
            
        try:
            ombro = (
                landmarks.landmark[self.LANDMARKS.RIGHT_SHOULDER.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_SHOULDER.value].y
            )
            cotovelo = (
                landmarks.landmark[self.LANDMARKS.RIGHT_ELBOW.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_ELBOW.value].y
            )
            punho = (
                landmarks.landmark[self.LANDMARKS.RIGHT_WRIST.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_WRIST.value].y
            )
            
            self.angulo_atual = calcular_angulo(ombro, cotovelo, punho)
            
            if self.angulo_atual < self.ANGULO_BAIXO and not self.em_baixo:
                self.em_baixo = True
                self.estado = "em_baixo"
                self.feedback = "Sobe!"
            elif self.angulo_atual > self.ANGULO_CIMA and self.em_baixo:
                self.repeticoes += 1
                self.em_baixo = False
                self.estado = "em_cima"
                self.feedback = f"Flexao {self.repeticoes}!"
                
        except Exception as e:
            self.feedback = f"Erro: {str(e)}"
            
        return self.repeticoes, self.estado, self.feedback
    
    def desenhar_info(self, imagem, landmarks):
        """SEM EMOJIS - apenas texto."""
        # Nome no canto superior direito
        h, w, _ = imagem.shape
        cv2.putText(imagem, f"{self.icone} {self.nome_display}", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cor, 2)
        
        # Angulo no cotovelo
        if hasattr(self, 'angulo_atual') and landmarks:
            try:
                cotovelo = landmarks.landmark[self.LANDMARKS.RIGHT_ELBOW.value]
                h, w, _ = imagem.shape
                pos = (int(cotovelo.x * w), int(cotovelo.y * h))
                cv2.putText(imagem, f"{self.angulo_atual:.0f}°", 
                           (pos[0]-30, pos[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cor, 2)
            except:
                pass
    
    @staticmethod
    def testar():
        """Testa flexao com webcam."""
        print("🧪 Testando Flexao... Pressiona 'q' para sair")
        import cv2
        from analisador.pose_detector import PoseDetector
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        detector = PoseDetector()
        exercicio = Flexao()
        
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
                
                cv2.putText(frame, f"Flexoes: {reps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, exercicio.cor, 2)
                cv2.putText(frame, feedback, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow("Teste Flexao", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Flexao.testar()