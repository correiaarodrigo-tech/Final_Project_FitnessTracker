"""
Funcoes auxiliares para desenhar informacoes na imagem.
"""

import cv2
import numpy as np

def desenhar_metricas(imagem_info, metricas):
    """
    Desenha metricas na janela de informacao.
    
    Args:
        imagem_info: Imagem da janela de info (geralmente preta)
        metricas: Dicionario com chave: valor
    """
    y = 30
    for chave, valor in metricas.items():
        texto = f"{chave}: {valor}"
        # Quebrar linha se muito longo
        if len(texto) > 30:
            texto = texto[:27] + "..."
            
        cv2.putText(imagem_info, texto, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        
        # Se passar do limite, comeca nova coluna
        if y > imagem_info.shape[0] - 50:
            y = 30
            # TODO: implementar segunda coluna se necessario

def criar_janela_info(largura=400, altura=300, titulo="Info"):
    """
    Cria uma imagem em branco para janela de informacao.
    
    Returns:
        numpy.ndarray: Imagem preta do tamanho especificado
    """
    return np.zeros((altura, largura, 3), dtype=np.uint8)

def desenhar_barra_progresso(imagem, progresso, posicao=(20, 200), largura=200, altura=20):
    """
    Desenha barra de progresso (0-100%).
    
    Args:
        imagem: Onde desenhar
        progresso: Valor 0-100
        posicao: (x, y) do canto superior esquerdo
        largura: Largura da barra
        altura: Altura da barra
    """
    x, y = posicao
    
    # Barra de fundo (cinza)
    cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (50, 50, 50), -1)
    
    # Barra de progresso (verde)
    preenchimento = int(largura * progresso / 100)
    cv2.rectangle(imagem, (x, y), (x + preenchimento, y + altura), (0, 255, 0), -1)
    
    # Contorno
    cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (255, 255, 255), 1)
    
    # Texto percentagem
    cv2.putText(imagem, f"{progresso:.0f}%", (x + largura + 10, y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === METODO DE TESTE ===
def testar():
    """Testa funcoes de visualizacao."""
    print("🧪 Testando visualizacao")
    
    img = criar_janela_info(500, 400)
    
    metricas = {
        "Exercicio": "Lunge",
        "Repeticoes": 8,
        "Estado": "em_baixo",
        "Feedback": "Perna direita a frente",
        "Pontuacao": 42
    }
    
    desenhar_metricas(img, metricas)
    desenhar_barra_progresso(img, 68, (20, 250))
    
    cv2.imshow("Teste Visualizacao", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    testar()