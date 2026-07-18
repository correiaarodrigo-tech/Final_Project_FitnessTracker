# Fundamentação Teórica: Estimativa de Pose via Inteligência Artificial

## 1. Descrição do Problema e Desafio
Manter a postura correta durante o exercício físico é fundamental para maximizar os benefícios do treino e prevenir lesões. Tradicionalmente, garantir uma postura correta exige a supervisão de um personal trainer ou fisioterapeuta. Contudo, com o aumento da popularidade dos treinos em casa, muitas pessoas exercitam-se sozinhas e sem supervisão.

**O Desafio:**
O principal desafio que este projeto visa resolver é fornecer monitorização e feedback em tempo real de forma precisa e acessível, sem exigir a aquisição de sensores dispendiosos, dispositivos *wearables* ou supervisão humana. Necessitamos de um sistema capaz de:
1. Reconhecer os movimentos corporais do utilizador em tempo real.
2. Calcular os ângulos das articulações para determinar se um exercício (como um agachamento ou flexão) está a ser executado corretamente.
3. Contar repetições de forma fiável.
4. Funcionar de forma eficiente em dispositivos comuns, como os smartphones.

## 2. Relevância do Projeto
Este projeto apresenta grande relevância no panorama atual da saúde e *fitness* digital:
*   **Acessibilidade:** Ao utilizar apenas a câmara padrão de um *smartphone*, democratiza o acesso a ferramentas de avaliação física.
*   **Prevenção de Lesões:** Ao alertar o utilizador para desvios na postura ideal, reduz substancialmente o risco de lesões musculoesqueléticas.
*   **Motivação e Acompanhamento:** A recolha de métricas (número de repetições, tempo sob tensão, cadência) ajuda a manter a motivação e permite registar a progressão temporal.

## 3. Arquitetura do MediaPipe e Edge Computing
A base tecnológica que torna esta aplicação possível em dispositivos móveis é a arquitetura avançada do *Google MediaPipe*. Para que o projeto funcione com sucesso em tempo real, é crucial compreender o seu fluxo de dados.

### 3.1. Processamento Assíncrono e Pacotes (*Packets*)
O *MediaPipe* não processa o vídeo como um bloco contínuo tradicional, operando através de um grafo de nós independentes (*calculators*).
*   **Timestamp e Sincronização:** A informação flui sob a forma de Pacotes (*Packets*). Cada pacote contém a *frame* de vídeo e um *Timestamp* numérico crescente. Este *timestamp* é a chave de sincronização do sistema.
*   **Natureza Assíncrona:** O sistema é descentralizado. Não existe um relógio global a bloquear o processamento; diferentes nós processam pacotes em simultâneo.

### 3.2. O *Scheduler* e a Gestão de *Frames*
Para avaliar a postura do utilizador, não precisamos de processar todas as 30 a 60 *frames* por segundo da câmara. O crucial é que a *frame* avaliada seja a mais recente, garantindo feedback sem latência perceptível.
*   **Descarte Inteligente (*Frame Dropping*):** O *MediaPipe* utiliza um *Scheduler* (gerido pelo `FlowLimiterCalculator`). Quando a câmara fornece *frames* a uma velocidade superior à capacidade de inferência da rede neuronal (*BlazePose*), o sistema descarta proativamente *frames* intermédias. Ao focar-se sempre na *frame* com o *timestamp* mais recente, sacrifica a fluidez visual da inferência em prol de **zero latência acumulada**.

### 3.3. Otimização de Processamento Local
A aplicação opera estritamente através de **Edge Computing**. Não existem chamadas a servidores *Cloud* para processamento de imagem, salvaguardando a privacidade e poupando bateria.
*   **Deteção e Rastreio:** O *BlazePose* aplica uma abordagem híbrida. Deteta a anatomia do corpo na primeira *frame* e, subsequentemente, apenas **rastreia** os 33 pontos-chave (*landmarks*), um processo muito mais leve computacionalmente.
*   **Cálculo Otimizado:** A nossa lógica Kotlin processa apenas matemática vetorial elementar (e.g., função inversa do cosseno `acos`), permitindo classificar o exercício numa fração de milissegundo após o *MediaPipe* devolver o *array* de coordenadas espaciais.

## 4. Avaliação e Pontuação Baseada em Limites Articulares
A comunidade científica de *Computer Vision* aplica métodos rígidos para avaliar a qualidade técnica de exercícios. Adotámos um método de **Métricas Cinemáticas (Ângulos Articulares) acopladas a uma Máquina de Estados Finita**.

Desconstruindo a **Flexão de Braços (Push-Up)**:

1.  **Fase Concêntrica / Descida ("DOWN"):** 
    *   **Métrica:** O ângulo interno Ombro-Cotovelo-Pulso tem de ser **inferior a 70 graus** (limite flexível dependendo do exercício).
    *   **Fundamento Científico:** Literatura (*IEEE*) sobre correção postural dita que uma flexão válida requer flexão do cotovelo até perto de $90^\circ$. Restringir a contagem a um *threshold* previne execuções curtas de baixa eficácia.
2.  **Fase Excêntrica / Subida ("UP"):** 
    *   **Métrica:** O ângulo tem de abrir para **mais de 150 graus**.
    *   **Fundamento Científico:** A extensão completa assinala a conclusão mecânica do movimento, compensando diferenças anatómicas onde o indivíduo não consiga realizar o bloqueio completo articular (*lockout* a $180^\circ$).
3.  **Análise Temporal e Contagem:**
    *   Um fator crucial na prevenção de falsos positivos (validações erróneas devido a trepidações na câmara) é a validação sequencial. O contador de repetições apenas avança se o *tracker* observar a transição imperativa do estado *DOWN* concluído de volta para o estado *UP*.

## 5. Estado da Arte e Referências
A literatura científica valida fortemente a metodologia utilizada, reforçando a aplicabilidade do *Edge Computing* em ambiente Android:

*   **"Real-Time Posture Monitoring for Effective Exercise Using MediaPipe"**: Este artigo analisa a mecânica de exercícios de membros inferiores (*Lunges* e *Squats*), corroborando que o cálculo articular via *MediaPipe* alcança precisões na ordem dos 86%, sendo altamente viável para dar *feedback* corretivo.
*   **"Real-Time Keypoint Based Pose Classification of Physical Exercises" (arXiv)**: Estudo que consolida o uso da rede *BlazePose* agregada a máquinas de estado locais em *smartphones*, comprovando a eficácia na separação de fases concêntricas e excêntricas.
*   **"Real-Time Digital Assistance for Exercise" (JOIV/IEEE)**: Investigação com foco em ambiente Android. Os autores provam que o poder computacional *mobile* contemporâneo dispensa servidores robustos, suportando plenamente a arquitetura de *Edge Computing* aqui adotada.
*   **"Real-Time Workout Posture Correction using OpenCV and MediaPipe"**: Fundamenta a imposição de *thresholds* fixos e lineares como ferramenta principal para gerar métricas de pontuação (*Scoring*), penalizando padrões lesivos de forma programática.
