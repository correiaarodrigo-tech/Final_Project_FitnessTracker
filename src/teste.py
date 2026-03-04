import cv2
import mediapipe as mp
import numpy as np

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Função para calcular ângulo (ex: cotovelo)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Variáveis para contagem
right_arm_count = 0
left_arm_count = 0
right_arm_down = False
left_arm_down = False
ANGLE_THRESHOLD = 90  # Ângulo considerado como "dobrado" (menor que este valor)

# Capturar vídeo da webcam com resolução maior
cap = cv2.VideoCapture(0)
# Aumentar a resolução da captura
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Largura maior (anterior era 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # Altura maior (anterior era 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obter dimensões do frame
    height, width, _ = frame.shape

    # Converter BGR (OpenCV) para RGB (MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Desenhar os landmarks no frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extrair coordenadas para o braço direito
        landmarks = results.pose_landmarks.landmark
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Extrair coordenadas para o braço esquerdo
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calcular ângulos
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Mostrar os ângulos no ecrã
        cv2.putText(frame, f"R:{int(right_angle)}", 
                   tuple(np.multiply(right_elbow, [width, height]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"L:{int(left_angle)}", 
                   tuple(np.multiply(left_elbow, [width, height]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Contagem para braço direito (quando dobra - ângulo diminui)
        if right_angle < ANGLE_THRESHOLD and not right_arm_down:
            right_arm_down = True
        elif right_angle >= ANGLE_THRESHOLD and right_arm_down:
            right_arm_down = False
            right_arm_count += 1

        # Contagem para braço esquerdo (quando dobra - ângulo diminui)
        if left_angle < ANGLE_THRESHOLD and not left_arm_down:
            left_arm_down = True
        elif left_angle >= ANGLE_THRESHOLD and left_arm_down:
            left_arm_down = False
            left_arm_count += 1

    # Criar overlay semi-transparente para o texto de fundo
    overlay = frame.copy()
    
    # Fundo para o texto de instrução (parte inferior)
    cv2.rectangle(overlay, (0, height-40), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Fundo para os contadores (parte superior)
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Adicionar texto de instrução no fundo (parte inferior)
    cv2.putText(frame, 'Clique na tecla Q para fechar a janela', 
                (int(width/2) - 250, height-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Adicionar contadores no topo
    # Contador direito (esquerda da tela)
    cv2.putText(frame, f'Braco Direito: {right_arm_count:3d}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    # Contador esquerdo (direita da tela) - calculado dinamicamente
    text_esquerdo = f'Braco Esquerdo: {left_arm_count:3d}'
    (text_width, text_height), _ = cv2.getTextSize(text_esquerdo, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pos_x_esquerdo = width - text_width - 20  # 20px da borda direita
    cv2.putText(frame, text_esquerdo, (pos_x_esquerdo, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    

    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()