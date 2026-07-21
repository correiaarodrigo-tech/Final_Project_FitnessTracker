"""
Wrapper para o MediaPipe Pose com funcionalidades de desenho e extração.
"""

import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    """
    Classe para detetar e desenhar landmarks do corpo usando MediaPipe.
    """
    
    def __init__(self, static_mode=False, min_detection=0.5, min_tracking=0.5):
        """
        Inicializa o detector de pose.
        
        Args:
            static_mode: Se True, processa cada frame como imagem estática (mais lento)
            min_detection: Confiança mínima para deteção inicial
            min_tracking: Confiança mínima para tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        
    def encontrar_landmarks(self, imagem):
        """
        Processa uma imagem e encontra os landmarks da pose.
        
        Args:
            imagem: Frame em BGR (formato OpenCV)
            
        Returns:
            landmarks ou None
        """
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imagem_rgb)
        return self.results.pose_landmarks if self.results else None
    
    def desenhar_landmarks(self, imagem, landmarks=None):
        """
        Desenha os landmarks e conexões na imagem.
        
        Args:
            imagem: Frame onde desenhar
            landmarks: Se None, usa os últimos resultados
        """
        if landmarks is None and self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks
            
        if landmarks:
            self.mp_draw.draw_landmarks(
                imagem, 
                landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
            )
    
    def extrair_ponto(self, landmarks, ponto_id):
        """
        Extrai coordenadas (x, y) normalizadas de um ponto específico.
        
        Args:
            landmarks: Landmarks do MediaPipe
            ponto_id: ID do ponto (ex: PoseLandmark.LEFT_KNEE.value)
            
        Returns:
            Tuple (x, y) normalizadas ou None
        """
        if landmarks and hasattr(landmarks, 'landmark') and ponto_id < len(landmarks.landmark):
            ponto = landmarks.landmark[ponto_id]
            return (ponto.x, ponto.y)
        return None
    
    # Teste local
    @staticmethod
    def testar():
        """
        Testa o PoseDetector com a webcam.
        Uso: python -m analisador.pose_detector
        """
        print("Testando PoseDetector... Pressiona 'q' para sair")
        cap = cv2.VideoCapture(0)
        detector = PoseDetector()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Encontrar e desenhar landmarks
            landmarks = detector.encontrar_landmarks(frame)
            detector.desenhar_landmarks(frame, landmarks)
            
            # Mostrar info
            if landmarks:
                cv2.putText(frame, "Pose detectada!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Teste PoseDetector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    PoseDetector.testar()