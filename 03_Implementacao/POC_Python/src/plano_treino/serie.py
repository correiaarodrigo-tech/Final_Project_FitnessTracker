"""
Gestão de séries de exercícios.
"""

import time

class Serie:
    """
    Representa uma sequência de exercícios com pausas.
    """
    
    def __init__(self, exercicios, pausa_entre_exercicios=10):
        """
        Args:
            exercicios: Lista de objetos ExercicioBase
            pausa_entre_exercicios: Segundos de pausa entre exercícios
        """
        self.exercicios = exercicios
        self.pausa = pausa_entre_exercicios
        self.indice_atual = 0
        self.em_pausa = False
        self.inicio_pausa = None
        self.tempo_restante_pausa = 0
        
    @property
    def exercicio_atual(self):
        """Retorna o exercício atual ou None se terminou."""
        if self.indice_atual < len(self.exercicios):
            return self.exercicios[self.indice_atual]
        return None
    
    def avancar(self):
        """Avança para o próximo exercício ou inicia pausa."""
        if self.indice_atual < len(self.exercicios) - 1:
            self.indice_atual += 1
            self.em_pausa = True
            self.inicio_pausa = time.time()
            return True
        return False  # Série terminada
    
    def atualizar_pausa(self):
        """Atualiza contador de pausa. Retorna True se pausa terminou."""
        if self.em_pausa and self.inicio_pausa:
            decorrido = time.time() - self.inicio_pausa
            self.tempo_restante_pausa = max(0, self.pausa - decorrido)
            
            if decorrido >= self.pausa:
                self.em_pausa = False
                self.inicio_pausa = None
                return True
        return False
    
    def reset(self):
        """Reinicia a série."""
        self.indice_atual = 0
        self.em_pausa = False
        self.inicio_pausa = None
        for ex in self.exercicios:
            ex.reset()
    
    # Teste local
    @staticmethod
    def testar():
        """Testa a lógica de séries."""
        print("Testando Serie (mock - sem câmara)")
        
        # Criar mocks de exercícios
        class MockExercicio:
            def __init__(self, nome):
                self.nome = nome
            def reset(self):
                print(f"  Reset {self.nome}")
            def __str__(self):
                return self.nome
        
        ex1 = MockExercicio("Agachamento")
        ex2 = MockExercicio("Prancha")
        ex3 = MockExercicio("Lunges")
        
        serie = Serie([ex1, ex2, ex3])
        
        print(f"Série com {len(serie.exercicios)} exercícios")
        print(f"Pausa de {serie.pausa}s entre exercícios")
        
        while serie.exercicio_atual:
            print(f"\nExercício atual: {serie.exercicio_atual}")

            # Simular fim do exercício
            input("Pressiona Enter quando terminares o exercício (mock)...")

            if serie.avancar():
                print("Pausa...")
                while serie.em_pausa:
                    serie.atualizar_pausa()
                    print(f"   Tempo restante: {serie.tempo_restante_pausa:.1f}s")
                    time.sleep(0.5)
                print("Próximo exercício!")
            else:
                print("Série concluída!")
                break
        print("\nFechando classe de teste!")

if __name__ == "__main__":
    Serie.testar()