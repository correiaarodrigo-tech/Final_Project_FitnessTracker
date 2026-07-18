# FitnessTracking: Aplicação Móvel para a Monitorização de Exercícios em Tempo Real

**Projeto de Final de Curso (LEIM) - Instituto Superior de Engenharia de Lisboa**

Uma aplicação móvel Android nativa projetada para atuar como um assistente de fitness inteligente. Utiliza *Edge Computing* (Google MediaPipe) para estimativa de pose (*Pose Estimation*) em tempo real, avaliando ângulos articulares, postura, e a correção biomecânica dos exercícios diretamente no telemóvel do utilizador. Sem recurso a processamento em servidores remotos (*Zero Latency*).

---

## 👥 Autores
- Rodrigo Correia (N.º 45155)
- David Delgado (N.º 51598)

---

## 📱 Apresentação (Funcionalidades em Vídeo)

Abaixo seguem os vídeos de demonstração da aplicação:

### Autenticação e Login
https://github.com/user-attachments/assets/9a9a8690-698d-446a-a72c-90059b276c34

### Dashboard e Hub do Utilizador
https://github.com/user-attachments/assets/677ec16e-456a-4229-849b-2f0a062bdfdf

### Dados do Utilizador e Edição de Perfil
https://github.com/user-attachments/assets/226ddb1f-8874-43c8-9075-0b979594e4b8

### Estatísticas do Utilizador (Leaderboards & Histórico)
https://github.com/user-attachments/assets/bd1314a2-cb20-4b94-8589-b98f078e5fce

### Calibração da Câmara e Ajuste de Distância
https://github.com/user-attachments/assets/a69d4c37-c23b-4934-8f15-7ea7e88a25d4

### Exercício de Agachamento (Squat) em Tempo Real (Feedback Visio-Acústico)
https://github.com/user-attachments/assets/42a02141-1ec1-4bf3-98a5-5dfbe93effd3

### Resultados do Exercício e Transação de Kcal/XP
https://github.com/user-attachments/assets/02eea182-ff08-405c-acd8-cbe90ddb5083

---

## ⚙️ Instalação e Utilização

### Pré-requisitos
*   **Hardware:** *Smartphone* Android com câmara funcional (mínimo recomendado para inferência em tempo real: processador Snapdragon série 700 ou equivalente).
*   **Software (Dispositivo):** Sistema operativo Android 8.0 (API Level 26) ou superior.
*   **Ambiente de Desenvolvimento:** Android Studio (Ladybug ou versão recente) para compilar a partir do código-fonte.

### Como Executar o Projeto
1.  **Clonar o Repositório:** Descarregue o código para o seu ambiente de desenvolvimento.
2.  **Abrir o Projeto:** No Android Studio, selecione `Open` e navegue para a pasta `03_Implementacao/AndroidApp_V_0.1/`.
3.  **Sincronização:** O *Gradle* irá automaticamente transferir todas as dependências (Jetpack Compose, Firebase BOM, MediaPipe Tasks Vision).
4.  **Compilar e Correr:** Ligue um dispositivo Android por cabo (com a Depuração USB ativada) ou inicie um emulador com suporte a câmara virtual e clique em `Run` (▶️).
5.  *(Alternativa)*: A aplicação também pode ser diretamente instalada através da geração e transferência do ficheiro APK (*Build > Build Bundle(s) / APK(s) > Build APK(s)*).

---

## 🚀 Trabalho Futuro

Como prova de conceito escalável, a arquitetura deste projeto foi desenhada para facilitar integrações avançadas que não fizeram parte da entrega MVP atual. (Nota: O trabalho futuro encontra-se dependente de um novo estudo de lançamento com novos requisitos funcionais e análise funcional, nos exatos mesmos moldes do projeto atual).

**1. Expansão da Biblioteca de Exercícios Multi-articulares**
*   **Exercícios Planeados:** *Overhead Press*, *Mountain Climbers*, *Bicep Curls* e *Jumping Jacks*. A lógica biomecânica já está parcialmente documentada, faltando apenas as suas classes ativas de rastreamento vetorial.

**2. Expansão de Funcionalidades de Utilizador para Maior Gamificação**
*   Implementação de mais campos editáveis para o utilizador, aproximando a aplicação de uma pequena rede social de *fitness*.
*   Suporte para foto de perfil (*Upload* via Firebase Storage).
*   Partilha externa dos resultados de treinos em outras redes sociais.

**3. Assistente de IA Generativa Local (Ollama)**
*   Ligação do histórico estruturado do Firestore a um modelo de linguagem local (e.g. LLama 3 ou Phi-3 via API Ollama). O assistente teria a capacidade de analisar os resultados agregados (ex: desvios na pontuação de forma) e ajudar o utilizador a definir objetivos diários, criar novos planos de treino corretivos e atuar como um Personal Trainer motivacional conversacional.
