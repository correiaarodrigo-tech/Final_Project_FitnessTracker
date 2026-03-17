"""
Ponto de entrada principal - Versao completa com janela extra e lunges.
Resolucao: 1280x720
"""

import cv2
import numpy as np
from analisador.pose_detector import PoseDetector
from exercicios.registry import ExercicioRegistry
from config.teclas import TeclaExercicio
from plano_treino.avaliador import Avaliador
from utils.visualizacao import desenhar_metricas, criar_janela_info

def configurar_camera():
    """Configura camera para 1280x720."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Verificar se conseguiu
    largura = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    altura = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📷 Camera configurada: {largura}x{altura}")
    
    return cap

def main():
    print("🏋️  Fitness Tracker v1.0 ")
    print("\nTeclas disponiveis:")
    ExercicioRegistry.listar_todos()
    print(f"\nPressiona: {TeclaExercicio.listar_disponiveis()}")
    print("  'q' - Sair\n")
    
    # Camera com resolucao definida
    cap = configurar_camera()
    detector = PoseDetector()
    
    # Comeca com agachamento
    exercicio_atual = ExercicioRegistry.get_exercicio(TeclaExercicio.AGACHAMENTO)
    avaliador = Avaliador()
    
    # Janela extra para metricas (mesmo tamanho da anterior)
    janela_info = criar_janela_info(400, 300)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Espelhar para ser mais intuitivo
        frame = cv2.flip(frame, 1)
        
        # Detectar pose
        landmarks = detector.encontrar_landmarks(frame)
        detector.desenhar_landmarks(frame, landmarks)
        
        # Limpar janela info (preto)
        janela_info[:] = 0
        
        # Processar se houver deteccao
        metricas = {}
        if landmarks and exercicio_atual:
            # Processar exercicio atual
            reps, estado, feedback = exercicio_atual.processar_landmarks(landmarks)
            exercicio_atual.desenhar_info(frame, landmarks)
            
            # Preparar metricas para janela extra
            metricas = {
                "Exercicio": exercicio_atual.nome_display,
                "Repeticoes": reps,
                "Estado": estado,
                "Feedback": feedback,
                "Pontuacao": avaliador.pontuacao_total
            }
            
            # Se for lunge, mostrar qual perna
            if exercicio_atual.nome_display == "Lunge" and hasattr(exercicio_atual, 'perna_ativa'):
                metricas["Perna ativa"] = exercicio_atual.perna_ativa
            
            # Avaliar (mock simples)
            if hasattr(exercicio_atual, 'angulo_atual'):
                pontos, _ = avaliador.avaliar_angulo(
                    exercicio_atual.angulo_atual, 90, tolerancia=20
                )
        else:
            metricas = {
                "Status": "Aguardando deteccao...",
                "Dica": "Posiciona-te na camera"
            }
        
        # Desenhar metricas na janela extra
        desenhar_metricas(janela_info, metricas)
        
        # Desenhar legenda de teclas no frame principal
        legenda = "Teclas: " + ' '.join([f"{t.value}:{t.name}" for t in TeclaExercicio])
        cv2.putText(frame, legenda, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Mostrar resolucao no canto (debug)
        cv2.putText(frame, "1280x720", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Mostrar ambas as janelas
        cv2.imshow("Fitness Tracker - Camera", frame)
        cv2.imshow("Informacao do Exercicio", janela_info)
        
        # Processar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key < 256:  # tecla valida
            tecla_char = chr(key)
            tecla_enum = TeclaExercicio.from_tecla(tecla_char)
            if tecla_enum:
                novo_exercicio = ExercicioRegistry.get_exercicio(tecla_enum)
                if novo_exercicio:
                    exercicio_atual = novo_exercicio
                    print(f"🔄 Mudou para {exercicio_atual.nome_display}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar resumo final
    print(f"\n📊 Sessao concluida!")
    print(f"🏆 Pontuacao final: {avaliador.pontuacao_total}")
    print(f"💬 {avaliador.get_classificacao()}")

if __name__ == "__main__":
    main()