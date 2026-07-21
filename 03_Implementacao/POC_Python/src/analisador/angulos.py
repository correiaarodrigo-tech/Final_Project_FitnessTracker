"""
Funções para cálculo de ângulos entre pontos do corpo.
"""

import numpy as np

def calcular_angulo(a, b, c):
    """
    Calcula o ângulo (em graus) entre os pontos a-b-c.
    
    Args:
        a, b, c: Tuplos (x, y) dos pontos (b é o vértice)
        
    Returns:
        Ângulo em graus (0-180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    vetor_ba = a - b
    vetor_bc = c - b
    
    # Produto escalar
    cos_angulo = np.dot(vetor_ba, vetor_bc) / (np.linalg.norm(vetor_ba) * np.linalg.norm(vetor_bc))
    # Evitar erros numéricos
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    
    angulo_rad = np.arccos(cos_angulo)
    angulo_graus = np.degrees(angulo_rad)
    
    return angulo_graus

def calcular_angulo_3d(a, b, c):
    """
    Versão 3D para maior precisão.
    
    Args:
        a, b, c: Tuplos (x, y, z)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    vetor_ba = a - b
    vetor_bc = c - b
    
    cos_angulo = np.dot(vetor_ba, vetor_bc) / (np.linalg.norm(vetor_ba) * np.linalg.norm(vetor_bc))
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angulo))

# Teste local
def testar():
    """Testa as funções de ângulo com valores conhecidos."""
    print("Testando calcular_angulo:")
    
    # Ângulo reto (90 graus)
    a = (0, 0)
    b = (0, 0)
    c = (0, 1)
    print(f"  Ponto a=b? -> {calcular_angulo(a, b, c):.1f}° (esperado 0°)")
    
    a = (0, 1)
    b = (0, 0)
    c = (1, 0)
    print(f"  Ângulo reto: {calcular_angulo(a, b, c):.1f}° (esperado 90°)")
    
    a = (0, 1)
    b = (0, 0)
    c = (0, -1)
    print(f"  Linha reta: {calcular_angulo(a, b, c):.1f}° (esperado 180°)")
    
    a = (1, 1)
    b = (0, 0)
    c = (1, 0)
    print(f"  Ângulo 45°: {calcular_angulo(a, b, c):.1f}° (esperado ~45°)")

if __name__ == "__main__":
    testar()