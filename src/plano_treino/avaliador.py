"""
Sistema simples de pontuação para exercícios.
"""

class Avaliador:
    """
    Avalia a qualidade da execução e atribui pontuação.
    """
    
    def __init__(self):
        self.pontuacao_total = 0
        self.historico = []
        
    def avaliar_angulo(self, angulo, esperado, tolerancia=10, pontos=1):
        """
        Avalia se um ângulo está dentro da tolerância.
        
        Args:
            angulo: Ângulo medido
            esperado: Ângulo ideal
            tolerancia: Tolerância em graus
            pontos: Pontos a atribuir se correto
            
        Returns:
            (pontos_ganhos, feedback)
        """
        if abs(angulo - esperado) <= tolerancia:
            self.pontuacao_total += pontos
            self.historico.append(("angulo", pontos))
            return pontos, "Posição correta!"
        else:
            return 0, "Ajusta a posição"
    
    def avaliar_tempo(self, tempo_mantido, tempo_objetivo, pontos_por_segundo=0.1):
        """
        Avalia tempo mantido numa posição.
        
        Args:
            tempo_mantido: Segundos na posição
            tempo_objetivo: Objetivo em segundos
            pontos_por_segundo: Pontos por segundo mantido
        """
        pontos = int(tempo_mantido * pontos_por_segundo)
        self.pontuacao_total += pontos
        self.historico.append(("tempo", pontos))
        
        progresso = (tempo_mantido / tempo_objetivo) * 100
        return pontos, f"Progresso: {progresso:.0f}%"
    
    def avaliar_repeticoes(self, reps, reps_objetivo, pontos_por_rep=10):
        """Avalia número de repetições."""
        pontos = reps * pontos_por_rep
        self.pontuacao_total += pontos
        self.historico.append(("repeticoes", pontos))
        
        progresso = (reps / reps_objetivo) * 100 if reps_objetivo > 0 else 0
        return pontos, f"Completaste {reps}/{reps_objetivo} reps"
    
    def reset(self):
        """Reinicia pontuação."""
        self.pontuacao_total = 0
        self.historico = []
    
    def get_classificacao(self):
        """Retorna classificação baseada na pontuação."""
        if self.pontuacao_total >= 100:
            return "💪 Excelente!"
        elif self.pontuacao_total >= 50:
            return "👍 Bom trabalho!"
        elif self.pontuacao_total >= 20:
            return "👌 Podes melhorar"
        else:
            return "🎯 Continua a tentar"
    
    # === MÉTODO DE TESTE LOCAL ===
    @staticmethod
    def testar():
        """Testa o avaliador com cenários simulados."""
        print("🧪 Testando Avaliador\n")
        
        avaliador = Avaliador()
        
        print("1. Avaliar ângulo (esperado 90°, tolerância 10°)")
        pontos, feedback = avaliador.avaliar_angulo(85, 90)
        print(f"   Ângulo 85° -> {feedback} (+{pontos} pts)")
        
        pontos, feedback = avaliador.avaliar_angulo(120, 90)
        print(f"   Ângulo 120° -> {feedback} (+{pontos} pts)")
        
        print("\n2. Avaliar tempo")
        pontos, feedback = avaliador.avaliar_tempo(15, 30)
        print(f"   {feedback} (+{pontos} pts)")
        
        print("\n3. Avaliar repetições")
        pontos, feedback = avaliador.avaliar_repeticoes(8, 10)
        print(f"   {feedback} (+{pontos} pts)")
        
        print(f"\n📊 Pontuação total: {avaliador.pontuacao_total}")
        print(f"🏆 Classificação: {avaliador.get_classificacao()}")
        
        print("\n📜 Histórico:")
        for tipo, pts in avaliador.historico:
            print(f"   - {tipo}: {pts} pts")

if __name__ == "__main__":
    Avaliador.testar()