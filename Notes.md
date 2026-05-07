#  Síntese

**Data:** Maio 2026  
**Autores:** Rodrigo Correia (45155), David Delgado (51598)

---

**Fontes:**
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Issue #4510 - Performance Android](https://github.com/ollama/ollama/issues/4510)
- [Google IDX Free Tier](https://idx.dev/docs)
- [LAMI Client - Exemplo Android](https://github.com/sonusid1325/ollama-android)

---


## 

**Funcionalidades possiveis:**
1. Feedback textual personalizado após cada série 
2. Criação de treinos por comando de voz?
3. Resumo motivacional pós-treino
4. Análise de dados de treino, planeamento regime treino

5. Mas principalmente, é uma ferramenta boa para montar encima das features desenvolvidas 'core' do projeto. NÃO as substitui.

** NÃO é possivel:**
- Feedback em tempo real via Ollama (MediaPipe no momento, Ollama por API calls)
- Correr Ollama locamente no telemóvel

---

## Fluxo da integração possivel
APLICAÇÃO ANDROID + MEDIAPIPE

 1. Utilizador completa série de 10 repetições

[MediaPipe] → deteta ângulos (joelho, anca) e ritmo

 2. Enviar dados para servidor Ollama

[App Android] → HTTP POST para http://<IP_SERVIDOR>:11434/api/generate
{
"model": "llama3.2:3b",
"prompt": "Ângulo joelho: 95°, ângulo anca: 130°, exercício: agachamento. Dá feedback de 1 frase."
}

 3. Servidor processa (2-5 segundos)

[Ollama na VM/PC] → gera resposta textual

 4. App recebe resposta

[App Android] → exibe feedback no ecrã durante pausa entre séries

 5. Guardar no histórico, mostrar texto ao utilizador

[Base de Dados] → armazena feedback com timestamp e série, apresneta feedback visual
