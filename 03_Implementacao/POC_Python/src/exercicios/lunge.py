"""
Implementacao do exercicio de lunge (avanco) com alternancia entre pernas.
"""

from exercicios.base_exercicio import ExercicioBase
from analisador.angulos import calcular_angulo
import cv2

class Lunge(ExercicioBase):
    """
    Deteta lunges alternados: perna direita e esquerda.
    Feedback indica qual perna avancou e recuou.
    """
    
    def __init__(self, nome_display="Lunge", cor=(255,0,255), icone="LUNGE"):
        super().__init__(nome_display, cor, icone)
        
        # Configuracoes especificas
        self.ANGULO_JOELHO_BAIXO = 80   # joelho traseiro perto do chao
        self.ANGULO_JOELHO_CIMA = 160   # perna estendida
        
        # Estado para cada perna
        self.perna_direita_baixo = False
        self.perna_esquerda_baixo = False
        self.perna_ativa = "nenhuma"    # direita, esquerda, nenhuma
        self.ultima_perna = None         # para garantir alternancia
        
        # Angulos atuais
        self.angulo_joelho_direito = 0
        self.angulo_joelho_esquerdo = 0
        
    def processar_landmarks(self, landmarks):
        """
        Processa landmarks para contar lunges alternados.
        Detecta qual perna esta a frente baseado na posicao x do pe.
        """
        if not landmarks:
            self.feedback = "Sem deteccao"
            return self.repeticoes, self.estado, self.feedback
            
        try:
            # Extrair pontos dos pes (para saber qual esta a frente)
            pe_direito_x = landmarks.landmark[self.LANDMARKS.RIGHT_FOOT_INDEX.value].x
            pe_esquerdo_x = landmarks.landmark[self.LANDMARKS.LEFT_FOOT_INDEX.value].x
            
            # Extrair pontos dos joelhos (para angulos)
            joelho_direito = (
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value].y
            )
            joelho_esquerdo = (
                landmarks.landmark[self.LANDMARKS.LEFT_KNEE.value].x,
                landmarks.landmark[self.LANDMARKS.LEFT_KNEE.value].y
            )
            
            # Para angulo do joelho direito (quadril-joelho-tornozelo)
            quadril_direito = (
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_HIP.value].y
            )
            tornozelo_direito = (
                landmarks.landmark[self.LANDMARKS.RIGHT_ANKLE.value].x,
                landmarks.landmark[self.LANDMARKS.RIGHT_ANKLE.value].y
            )
            
            # Para angulo do joelho esquerdo
            quadril_esquerdo = (
                landmarks.landmark[self.LANDMARKS.LEFT_HIP.value].x,
                landmarks.landmark[self.LANDMARKS.LEFT_HIP.value].y
            )
            tornozelo_esquerdo = (
                landmarks.landmark[self.LANDMARKS.LEFT_ANKLE.value].x,
                landmarks.landmark[self.LANDMARKS.LEFT_ANKLE.value].y
            )
            
            # Calcular angulos
            self.angulo_joelho_direito = calcular_angulo(quadril_direito, joelho_direito, tornozelo_direito)
            self.angulo_joelho_esquerdo = calcular_angulo(quadril_esquerdo, joelho_esquerdo, tornozelo_esquerdo)
            
            # Determinar qual perna esta a frente (menor x = mais a direita na imagem)
            # Como a imagem esta espelhada, inverte a logica
            if pe_direito_x < pe_esquerdo_x:  # pe direito aparece mais a direita
                perna_frente = "direita"
                angulo_frente = self.angulo_joelho_direito
                angulo_tras = self.angulo_joelho_esquerdo
            else:
                perna_frente = "esquerda"
                angulo_frente = self.angulo_joelho_esquerdo
                angulo_tras = self.angulo_joelho_direito
            
            self.perna_ativa = perna_frente
            
            # Logica de contagem baseada no joelho de tras (o que vai ao chao)
            # Para lunge, o joelho traseiro e que fleete
            if perna_frente == "direita":
                # Joelho esquerdo e o traseiro
                if self.angulo_joelho_esquerdo < self.ANGULO_JOELHO_BAIXO and not self.perna_esquerda_baixo:
                    self.perna_esquerda_baixo = True
                    self.estado = "em_baixo"
                    self.feedback = f"Perna {perna_frente} a frente - Desce mais!"
                elif self.angulo_joelho_esquerdo > self.ANGULO_JOELHO_CIMA and self.perna_esquerda_baixo:
                    # Completou movimento
                    if self.ultima_perna != perna_frente:  # alternou?
                        self.repeticoes += 1
                        self.ultima_perna = perna_frente
                        self.feedback = f"Lunge {self.repeticoes} - Perna {perna_frente}!"
                    self.perna_esquerda_baixo = False
                    self.estado = "em_cima"
            else:  # perna esquerda a frente
                # Joelho direito e o traseiro
                if self.angulo_joelho_direito < self.ANGULO_JOELHO_BAIXO and not self.perna_direita_baixo:
                    self.perna_direita_baixo = True
                    self.estado = "em_baixo"
                    self.feedback = f"Perna {perna_frente} a frente - Desce mais!"
                elif self.angulo_joelho_direito > self.ANGULO_JOELHO_CIMA and self.perna_direita_baixo:
                    # Completou movimento
                    if self.ultima_perna != perna_frente:  # alternou?
                        self.repeticoes += 1
                        self.ultima_perna = perna_frente
                        self.feedback = f"Lunge {self.repeticoes} - Perna {perna_frente}!"
                    self.perna_direita_baixo = False
                    self.estado = "em_cima"
            
        except Exception as e:
            self.feedback = f"Erro: {str(e)}"
            
        return self.repeticoes, self.estado, self.feedback
    
    def desenhar_info(self, imagem, landmarks):
        """Desenha informacoes especificas do lunge."""
        # Nome no canto superior direito
        cv2.putText(imagem, f"LUNGE", (imagem.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cor, 2)
        
        # Mostrar perna ativa
        if hasattr(self, 'perna_ativa'):
            cor_perna = (0, 255, 0) if self.perna_ativa == "direita" else (255, 255, 0)
            texto = f"Perna: {self.perna_ativa}"
            cv2.putText(imagem, texto, (imagem.shape[1] - 250, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_perna, 2)
        
        # Desenhar angulos dos joelhos
        if landmarks:
            try:
                # Angulo joelho direito
                joelho_d = landmarks.landmark[self.LANDMARKS.RIGHT_KNEE.value]
                h, w, _ = imagem.shape
                pos_d = (int(joelho_d.x * w), int(joelho_d.y * h))
                cv2.putText(imagem, f"D:{self.angulo_joelho_direito:.0f}", 
                           (pos_d[0]-40, pos_d[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                
                # Angulo joelho esquerdo
                joelho_e = landmarks.landmark[self.LANDMARKS.LEFT_KNEE.value]
                pos_e = (int(joelho_e.x * w), int(joelho_e.y * h))
                cv2.putText(imagem, f"E:{self.angulo_joelho_esquerdo:.0f}", 
                           (pos_e[0]-40, pos_e[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            except:
                pass
    
    @staticmethod
    def testar():
        """Testa lunge com webcam."""
        print("Testando Lunge... Pressiona 'q' para sair")
        import cv2
        from analisador.pose_detector import PoseDetector
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        detector = PoseDetector()
        exercicio = Lunge()
        
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
                
                cv2.putText(frame, f"Lunges: {reps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, exercicio.cor, 2)
                cv2.putText(frame, feedback, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, f"Estado: {estado}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            cv2.imshow("Teste Lunge", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Lunge.testar()