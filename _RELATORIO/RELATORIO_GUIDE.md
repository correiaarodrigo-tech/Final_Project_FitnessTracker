# Guia de Estrutura do Relatório (Estrutura Linear)

Este documento serve como mapa de planeamento para redigir o relatório oficial do projeto, seguindo a **Estrutura Linear (Usual)** definida pelo ISEL (modelo "book" em LaTeX).

A Estrutura Linear caracteriza-se por uma narrativa contínua, coerência global e uma extensão máxima estrita de **40 páginas** (excluindo capa, índices e anexos). Abaixo encontra-se a distribuição otimizada das páginas por capítulo para acomodar as necessidades específicas do projeto Fitness Tracker.

---

## Estrutura Otimizada (Extensão Total: ~35 a 40 Páginas)

### Capítulo 1: Introdução (~3 Páginas)
*   **Contexto e Motivação:** A crescente adoção da prática de exercício físico no domicílio e a democratização da Inteligência Artificial associada ao paradigma de *Edge Computing* (processamento distribuído em dispositivos móveis).
*   **Formulação do Problema:** A inerente dificuldade na manutenção da postura biomecânica correta e na contabilização rigorosa de repetições na ausência de supervisão profissional.
*   **Objetivos e Contributos:** Conceção e desenvolvimento de uma aplicação móvel nativa (Android) capaz de extrair o mapeamento tridimensional da pose humana em tempo real, fornecendo correção postural, contagem autónoma de repetições e mecânicas de gamificação. Dever-se-á referenciar o motor de avaliação, a máquina de estados implementada e o sistema de sintetização de voz (TTS).
*   **Estrutura do Documento:** Síntese da organização temática dos capítulos subsequentes.

### Capítulo 2: Trabalho Relacionado (~5 Páginas)
*   **Sensores Inerciais (IMUs) vs. Visão por Computador:** Estudo comparativo das abordagens clássicas e contemporâneas no rastreio da atividade física.
*   **Frameworks de Estimativa de Pose:** Análise crítica das arquiteturas *OpenPose*, *YOLO-pose* e *MediaPipe BlazePose*. 
*   **Privacidade e Edge Computing:** Fundamentação da escolha do *MediaPipe* justificada pela viabilidade do processamento inferencial local (*Edge AI*), assegurando baixa latência e a retenção de dados sensíveis no dispositivo do utilizador em detrimento de soluções assentes em servidores remotos.
*   **Aplicações Similares na Literatura:** Revisão bibliográfica de soluções análogas orientadas à deteção e avaliação motora.

### Capítulo 3: Modelo Proposto (~8 Páginas)
*   **Requisitos do Sistema:** Elicitação dos requisitos funcionais e não-funcionais, com especial enfoque na autonomia do utilizador.
*   **Fundamentos e Cinesiologia:** Definição formal dos eixos articulares e da respetiva modelação matemática (álgebra vetorial e aplicação da função \(\arccos\)).
*   **Amplitudes de Movimento (ROM):** Formalização da teoria biomecânica determinante para o sucesso da execução motora (baseada nos postulados de Norkin e White) nos exercícios: *Squats*, *Push-ups*, *Lunges* e *Plank*.
*   **Métrica de Avaliação (Form Scoring):** Conceptualização teórica da avaliação do desempenho, estipulando os critérios matemáticos de penalização perante desvios angulares, limites isométricos e dissonância cadencial.

### Capítulo 4: Implementação do Modelo (~12 Páginas)
*   **Arquitetura Android:** Adoção do padrão de desenho MVVM (*Model-View-ViewModel*), injeção de dependências e a interface de comunicação da biblioteca *CameraX* com o módulo *PoseLandmarker* do *MediaPipe*.
*   **A Máquina de Estados:** Representação e descrição lógica do autómato finito responsável pela validação das fases do exercício (\texttt{AT\_TOP} $\to$ \texttt{DESCENDING} $\to$ \texttt{ASCENDING} $\to$ \texttt{AT\_TOP}). 
*   **A Mecânica de Oclusão (O caso Lunge):** Evidência da estratégia algorítmica adotada (rastreio do joelho em posição frontal) para mitigar a latência e falsos negativos decorrentes da oclusão visual.
*   **Feedback Corretivo Nativo (TTS):** Metodologia de atuação do sistema *Text-To-Speech* em tempo real, com ênfase na implementação do mecanismo de *debounce* (atraso intencional de 500ms) para suprimir a sobreposição acústica.
*   **Gamificação e Arquitetura Firestore:** Descrição da camada analítica e respetiva gestão assíncrona de dados via transações.

### Capítulo 5: Validação e Testes Práticos (~8 Páginas)
*   **Ambientes de Teste de Performance:** Validação experimental da *pipeline* de IA, contrapondo o comportamento em arquiteturas ARM (físicas) face a constrangimentos emulados (x86).
*   **Testes de Usabilidade (SUS):** 
    *   Análise quantitativa do formulário *System Usability Scale* segregada por estratos de literacia tecnológica.
    *   Ações Corretivas Implementadas: Localização integral da interface para a língua Portuguesa (focada no Grupo A) e redimensionamento tipográfico/vetorial do ecrã para visibilidade à distância ótima de enquadramento (2-6 metros).
*   **Integração IA e Relatório (Metodologia):** Clarificação metodológica da inclusão do ficheiro `prompt_set.txt` e uso da anotação `#my_code` para certificação da autoria algorítmica.

### Capítulo 6: Conclusões e Trabalho Futuro (~2 Páginas)
*   **Síntese Final:** Reflexão sobre a proficiência dos dispositivos móveis contemporâneos enquanto ferramentas biométricas viáveis e a consolidação do modelo de *Edge Computing*.
*   **Trabalho Futuro:** Prospetiva de expansões funcionais (e.g., sistemas multijogador síncronos, interconectividade com *smartwatches* e rastreio volumétrico da rotação da pélvis para refinação da análise lombar).

---

## Dicas para Redação Linear
- Certifique-se de que cada capítulo termina com uma breve "ponte" de transição para o capítulo seguinte, garantindo a "coerência global" exigida pela estrutura.
- Todo o código crítico deve ser colocado em caixas "Listing" formatadas ou referenciado (com a respetiva etiqueta `#my_code` num anexo).
