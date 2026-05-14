# Rastreador de Fitness com Estimativa de Pose via IA: Fundamentação e Pesquisa

## 1. Descrição do Problema e Desafio
Manter a postura correta durante o exercício é fundamental para maximizar os benefícios do treino e, mais importante ainda, para prevenir lesões. Tradicionalmente, garantir uma postura correta exige a presença de um personal trainer ou fisioterapeuta. Contudo, com o aumento da popularidade dos treinos em casa, muitas pessoas exercitam-se sozinhas e sem supervisão profissional.

**O Desafio:**
O principal desafio que este projeto visa resolver é como fornecer monitorização e feedback em tempo real, de forma precisa e acessível, sem exigir a compra de sensores caros, dispositivos vestíveis (wearables) ou supervisão humana. Necessitamos de um sistema capaz de:
1. Reconhecer os movimentos corporais do utilizador em tempo real.
2. Calcular os ângulos das articulações para determinar se um exercício (como um agachamento ou flexão) está a ser executado corretamente.
3. Contar repetições de forma fiável.
4. Funcionar de forma eficiente em dispositivos comuns, como os smartphones.

## 2. Relevância do Projeto
Este projeto tem uma enorme relevância no panorama atual da saúde e fitness digital:
*   **Acessibilidade:** Ao utilizar apenas a câmara padrão do smartphone, democratiza o acesso a ferramentas de avaliação física.
*   **Prevenção de Lesões:** Ao alertar o utilizador para desvios na postura ideal, reduz substancialmente o risco de lesões musculoesqueléticas.
*   **Motivação e Acompanhamento:** A recolha de métricas (número de repetições, tempo sob tensão, cadência) ajuda a manter a motivação e permite registar a progressão ao longo do tempo.

## 3. Arquitetura do MediaPipe e o Cerne da Lógica de Negócio
A base tecnológica que torna esta aplicação possível em dispositivos móveis é a arquitetura avançada do MediaPipe. Para que o projeto funcione com sucesso em *real-time* (tempo real), é crucial compreender como o MediaPipe gere o fluxo de dados.

### 3.1. Processamento Assíncrono e Pacotes (Packets)
O MediaPipe não processa vídeo como um bloco contínuo tradicional, mas sim através de um grafo de nós independentes (calculators). 
*   **Par Frame-Timestamp:** A informação flui sob a forma de "Pacotes" (*Packets*). Cada pacote contém a carga útil (a frame de vídeo da câmara) e um **Timestamp** (carimbo de tempo) numérico crescente. Este timestamp é a chave de sincronização de todo o sistema.
*   **Natureza Assíncrona:** O sistema é descentralizado. Não existe um relógio global a bloquear o processamento; diferentes nós podem processar pacotes de diferentes timestamps em simultâneo.

### 3.2. O Scheduler e a Gestão de Frames (Descarte)
**A Lógica de Negócio do Projeto:** Para avaliar se um utilizador fez um agachamento, **não precisamos de processar todas as 30 ou 60 frames por segundo que a câmara capta**. O importante é ter a certeza de que a frame processada é a mais *recente* possível, de modo a captar o movimento (a mudança de ângulos). O atraso (*latency*) é o maior inimigo do nosso projeto; processar frames antigas resultaria num feedback desfasado da realidade do utilizador.

*   **Descarte Inteligente (Frame Dropping):** O MediaPipe utiliza um *Scheduler* (gerido frequentemente pelo `FlowLimiterCalculator`). Quando o sistema deteta que a câmara está a enviar frames mais depressa do que o telemóvel consegue calcular a rede neural (BlazePose), ele **descarta proativamente as frames intermédias**. 
*   Ao focar-se apenas na frame do topo da fila (com o timestamp mais recente), o MediaPipe garante que a pose devolvida corresponde exatamente ao que o utilizador está a fazer naquele milissegundo, sacrificando frames intermédias em prol de **zero latência acumulada**.

### 3.3. Otimizações: O que o MediaPipe faz vs. O que nós fazemos
*   **O que o MediaPipe já otimiza:** Em vez de correr a pesada deteção de corpo inteiro em todas as frames, o BlazePose usa uma abordagem de "Deteção e Rastreio". Deteta o corpo na primeira frame e, nas frames seguintes, apenas **rastreia** os 33 pontos, o que é computacionalmente muito mais leve.
*   **O que nós podemos fazer (Boas Práticas):** 
    1. **Reduzir a Resolução de Entrada:** Enviar frames para o MediaPipe com resoluções menores (ex: 480p ou 720p em vez de 4K), pois a rede neural interna redimensiona a imagem de qualquer forma.
    2. **Lógica Eficiente (O nosso código Kotlin):** A nossa máquina de estados e o `AngleCalculator` utilizam matemática vetorial simples e leve (como o cálculo do cosseno com `acos`), permitindo que a avaliação do exercício aconteça numa fração de milissegundo logo após o MediaPipe nos devolver os landmarks.

## 4. Avaliação e Pontuação de Exercícios (Desconstruindo a Flexão)
A comunidade científica de *Computer Vision* aplica métodos estritos para pontuar (avaliar a qualidade) e classificar exercícios. Na nossa aplicação, utilizamos o método de **Métricas Cinemáticas (Ângulos Articulares) acopladas a uma Máquina de Estados**. 

Para exemplificar, vamos desconstruir a lógica usada na nossa implementação da **Flexão (Push-Up)** (`PushUpExercise.kt`), sustentada por práticas académicas:

### 4.1. O que é analisado? (Pontos de Referência)
A literatura indica que a avaliação de um exercício requer a seleção de um subconjunto de pontos-chave específicos (*landmarks*) que representam a biomecânica principal do movimento.
Para a flexão, utilizamos um vetor tridimensional no braço, definido por:
*   **Ombro (Shoulder)** - Ponto 12 no MediaPipe.
*   **Cotovelo (Elbow)** - Ponto 14. (Atua como o vértice do ângulo).
*   **Pulso (Wrist)** - Ponto 16.

*(Nota: Embora estudos mais complexos também avaliem o alinhamento da coluna usando os pontos da anca e tornozelo para dar um "Score de Postura", o indicador principal de execução de uma repetição recai no grau de extensão e flexão do braço).*

### 4.2. Como pontuamos e avaliamos? (Amplitude de Movimento - ROM)
Os estudos demonstram que, para avaliar a correta execução, os sistemas definem *thresholds* (limites) baseados no modelo ideal humano (*Ideal Model Comparison*). A nossa avaliação baseia-se numa máquina de estados sequencial:

1.  **Fase Concêntrica / Descida ("DOWN"):** 
    *   **A nossa métrica:** O ângulo formado pelo Ombro-Cotovelo-Pulso tem de ser **inferior a 70 graus**.
    *   **Base científica:** Estudos como o publicado na *IEEE* sobre *Real-Time Workout Posture Correction* definem que uma flexão validada ("well done") exige que o cotovelo atinja pelo menos 90 graus de flexão. O nosso limite rigoroso de < 70º garante que o utilizador está a executar a **Amplitude de Movimento (ROM - Range of Motion)** completa, não permitindo "meias repetições".
2.  **Fase Excêntrica / Subida ("UP"):** 
    *   **A nossa métrica:** O ângulo tem de abrir para **mais de 150 graus**.
    *   **Base científica:** A extensão máxima do braço humano aproxima-se dos 180 graus. Fixar o threshold nos > 150º compensa ocos (soft-lockouts) anatómicos normais e variações na posição da câmara, garantindo que o movimento foi concluído de forma segura e não hiperestendida.
3.  **Contagem e Prevenção de Fraude (Temporal Analysis):**
    *   Um ponto fulcral referenciado em *papers* sobre contagem de repetições é o uso de validação sequencial. A repetição (`repetitions++`) **apenas** é contabilizada se o sistema detetar a transição correta do estado de *Down* (isDown = true) para *Up*. Isto evita avaliações falsas positivas derivadas de "saltos" da câmara.

## 5. Pesquisa Académica e Estado da Arte (Referências)
A literatura científica valida fortemente a nossa abordagem metodológica, com especial ênfase na viabilidade de implementar estes sistemas em plataformas mobile (Android) e na análise de exercícios mais complexos como Afundos (Lunges):

*   **"Real-Time Posture Monitoring for Effective Exercise Using MediaPipe":** Este artigo foca-se na análise rigorosa de exercícios de pernas, em particular os **Lunges (Afundos)**. Os autores comprovam que o cálculo de ângulos (quadril, joelho, tornozelo) através dos landmarks do MediaPipe atinge **cerca de 86% de precisão** na deteção da postura correta de um Lunge, validando a fiabilidade da ferramenta para dar feedback corretivo.
*   **"Real-Time Keypoint Based Pose Classification of Physical Exercises" (arXiv):** Estudo que consolida a arquitetura de extrair pontos com a rede BlazePose (a mesma que usamos) e canalizar esses dados num telemóvel para redes neuronais (LSTM) ou máquinas de estados. Demonstram uma elevada precisão na diferenciação entre as fases ativas e passivas em Flexões, Agachamentos e Lunges.
*   **"Real-Time Digital Assistance for Exercise" (JOIV/IEEE):** Investigação focada na **implementação Android**. Os autores sublinham que a arquitetura leve do BlazePose elimina a necessidade de processamento em servidores externos pesados. O telemóvel Android consegue, autonomamente e em tempo real, inferir os movimentos do utilizador e avaliar a biomecânica sem atrasos percetíveis, validando exatamente a arquitetura (Edge Computing) que adotámos neste projeto Android.
*   **"Real-Time Workout Posture Correction using OpenCV and MediaPipe":** Fundamenta a utilização de *thresholds* angulares como a principal fonte de "Scoring". Eles demonstram que um exercício executado dentro dos limites ideais aumenta a precisão da contagem e previne padrões lesivos.

## 6. Implementações Futuras (Roadmap)
Atualmente o núcleo (MediaPipe + Lógica em Kotlin) está funcional. No futuro, o projeto irá englobar:
*   **Avaliação Postural Completa:** Adicionar um cálculo simultâneo do alinhamento Ombro-Anca-Tornozelo durante a flexão. Se a coluna "cair" (ângulo inferior a 160º), o sistema não contabiliza a repetição e envia o feedback: "Endireite as costas!".
*   **Painel de Analítica:** Gráficos do histórico e gasto calórico (kcal).
*   **Autenticação de Utilizadores:** Sistemas de login seguros (ex: Firebase).
