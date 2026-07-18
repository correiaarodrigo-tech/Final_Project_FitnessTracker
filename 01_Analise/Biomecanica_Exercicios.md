# Fundamentação Biomecânica dos Exercícios (Limites Articulares e Pontuação)

Para garantir rigor científico e eficácia clínica, a estimativa de pose (*Pose Estimation*) e o motor de pontuação da nossa aplicação baseiam-se num cruzamento analítico de três grandes fontes de autoridade no treino e biomecânica desportiva:

1.  **ACSM (American College of Sports Medicine):** *Guidelines for Exercise Testing and Prescription* e *ACSM's Resources for the Personal Trainer*. Estabelecem os protocolos padrão de resistência muscular, amplitudes de movimento (*Range of Motion - ROM*) clínicas e os limites articulares seguros.
2.  **NSCA (National Strength and Conditioning Association):** *Essentials of Strength Training and Conditioning*. Define detalhadamente a cinemática articular, os ângulos ótimos de contração muscular e o alinhamento da coluna/tronco na estabilização do core.
3.  **FITescola / DGE (Direção-Geral da Educação, Portugal):** *Manual de Testes da Aptidão Física FITescola*. Serve como a referência principal da Educação Física em Portugal para padronização da execução de flexões e cadências com avaliação de falha técnica.
4.  **ACE (American Council on Exercise):** *ACE Personal Trainer Manual*. Orienta sobre a estruturação dos 5 movimentos fundamentais (agachar, afundar, empurrar, puxar e rodar) e os limites fisiológicos da estabilização isométrica.

Abaixo estão detalhados os parâmetros cinemáticos e os *landmarks* vetoriais do MediaPipe para os exercícios implementados e em fase de planeamento:

---

## 1. Exercícios de Tronco e Membros Superiores

### 1.1. Flexão de Braços (Push-Up) — `[Implementado]`
*   **Protocolo:** Posição inicial de prancha alta com os braços totalmente esticados. Descida controlada do tronco em linha reta até o peito aproximar-se do solo. Subida explosiva até à extensão dos cotovelos sem hiperextensão articular.
*   **Articulação Chave e Landmarks:** Cotovelo. Vetor gerado entre: Ombro [11/12] - Cotovelo [13/14, vértice] - Pulso [15/16].
*   **Ângulo DOWN (Flexão):** $\theta \le 70^\circ - 90^\circ$ (braço superior paralelo ao solo).
*   **Ângulo UP (Extensão):** $\theta \ge 150^\circ$.
*   **Alinhamento da Coluna:** O ângulo Ombro-Anca-Tornozelo ideal $\approx 180^\circ \pm 15^\circ$ para evitar a anca descaída (pressão lombar) ou elevada (compensação com os deltóides).
*   **Fontes de Validação:**
    *   *ACSM (Push-up Endurance Assessment):* Exige flexão completa até ao ângulo reto ($90^\circ$) no cotovelo ou contacto no solo.
    *   *FITescola:* Obriga o corpo a estar rigidamente alinhado numa linha reta, acompanhando uma cadência acústica intermitente.

### 1.2. Prancha Isométrica Frontal (Plank) — `[Implementado]`
*   **Protocolo:** Corpo assente nos antebraços e pontas dos pés. Corpo estático, resistente e rígido ao longo do eixo horizontal, suportando a força gravitacional isométricamente.
*   **Métrica Principal:** Alinhamento corporal (Landmarks 11/12, 23/24, 27/28).
*   **Limites de Postura:** Ângulo global de $\approx 180^\circ \pm 15^\circ$ (aceitável entre $165^\circ$ e $195^\circ$). Desvios fora deste intervalo acionam um alerta imediato de colapso ou elevação da postura (falha isómetrica).
*   **Fontes de Validação:**
    *   *NSCA & ACE (Isometric Core Assessment):* Avaliação rigorosa da resistência estática do *core* prevenindo a fadiga que habitualmente resulta numa cifose lombar ou hiperlordose da bacia.

### 1.3. Overhead Press — `[Trabalho Futuro]`
*   **Protocolo:** Movimento concêntrico vertical. O utilizador empurra o peso desde os ombros até atingir a extensão completa dos braços por cima da cabeça.
*   **Articulação Chave:** Cotovelos e Ombros.
*   **Ângulo Inicial (DOWN):** Cotovelos dobrados a $\le 90^\circ$ junto aos ombros.
*   **Ângulo Final (UP):** Extensão total dos braços num formato "*overhead*" ($\ge 165^\circ$).
*   **Fontes de Validação:**
    *   *NSCA (Vertical Push):* Destaca a crucial estabilidade da coluna em carga e a simetria bilateral de movimento.

### 1.4. Flexão de Bíceps (Bicep Curl) — `[Trabalho Futuro]`
*   **Protocolo:** Braço colado lateralmente ao tronco. Realiza-se a flexão puramente isolada da articulação do cotovelo, limitando inércias indesejadas.
*   **Articulação Chave:** Cotovelo (Ombro [11/12] - Cotovelo [13/14] - Pulso [15/16]).
*   **Ângulo DOWN:** $\theta \ge 160^\circ$ (braço pendente estendido).
*   **Ângulo UP:** $\theta \le 45^\circ$ (tensão máxima do bícipete).
*   **Fontes de Validação:**
    *   *NSCA (Single-Joint Arm Kinetics):* Regula as proibições do balanço compensatório da articulação do ombro.

---

## 2. Exercícios de Membros Inferiores

### 2.1. Agachamento (Squat) — `[Implementado]`
*   **Protocolo:** Posição de pé (ortostática). Executa-se o afundamento da bacia num movimento pélvico para trás, dobrando as pernas, com a subsequente subida usando a força das pernas contra o chão.
*   **Articulação Chave:** Joelho. Vetor gerado entre: Anca [23/24] - Joelho [25/26, vértice] - Tornozelo [27/28].
*   **Ângulo DOWN (Flexão):** $\theta \le 70^\circ - 90^\circ$ (coxas perfeitamente paralelas ao solo).
*   **Ângulo UP (Extensão):** $\theta \ge 150^\circ - 160^\circ$ (fase vertical ortostática).
*   **Alinhamento do Tronco:** A inclinação fisiológica do eixo vertebral (Ombro-Anca) versus o eixo vertical não deve exceder $45^\circ$ para evitar rotura discal.
*   **Fontes de Validação:**
    *   *NSCA (Squat Biomechanics):* Exige a flexão paralela ($70^\circ$-$90^\circ$) sem desvio patelar em valgo (sem torcer os joelhos para dentro do eixo natural).

### 2.2. Afundo (Lunge) — `[Implementado]`
*   **Protocolo:** Passo extenso em frente. O eixo da anca desce de forma estritamente vertical até à fase final do movimento.
*   **Articulação Chave:** Ambos os Joelhos e Anca.
*   **Ângulos de Flexão (DOWN):** O joelho da perna ativa anterior deve flexionar até $\le 80^\circ - 90^\circ$. Simultaneamente, o joelho posterior deve alcançar $\le 80^\circ - 90^\circ$, operando acima do plano do solo.
*   **Alinhamento do Tronco:** Vetor da coluna mantido verticalmente constante (desvio máximo da Anca-Ombro em relação à vertical $\le 15^\circ$).
*   **Fontes de Validação:**
    *   *ACE (Single-Leg Pattern):* Joelhos operam e absorvem a carga num limite angular rigoroso de $90^\circ$ alinhados transversalmente à ação gravitacional.

---

## 3. Exercícios Cardiovasculares (Cardio e Metcon)

### 3.1. Mountain Climbers — `[Trabalho Futuro]`
*   **Protocolo:** Posição de partida fixa em prancha alta. As pernas ciclam, trazendo de forma vigorosa e alternada o joelho fletido de encontro à caixa torácica.
*   **Métricas Dinâmicas:** A perna de apoio estabiliza o alinhamento corporal estático num limiar de $\theta \approx 180^\circ \pm 15^\circ$, enquanto a perna transiente flete agudamente o joelho ($\le 70^\circ$).
*   **Fontes de Validação:**
    *   *ACE (Dynamic Core Stabilization):* Exige uma enorme resistência anti-rotacional do tronco à fadiga imposta pela ciclagem motora.

### 3.2. Jumping Jacks — `[Trabalho Futuro]`
*   **Protocolo:** Atividade explosiva baseada em saltos laterais e afastamentos simétricos das pernas acoplados ao fecho de mãos acima da cabeça.
*   **Métricas de Abertura (Fase OUT):** Distância linear entre os Tornozelos (27 e 28) $> 1.5 \times$ largura anatómica dos ombros. Mãos circundam num ângulo onde a altura dos Pulsos (eixo Y) ultrapassa os Ombros.
*   **Métricas de Retorno (Fase IN):** Os tornozelos realinham em proximidade e as mãos regressam passivamente ao eixo inferior da anca.
*   **Fontes de Validação:**
    *   *ACSM (Cardiorespiratory Warm-up):* Foca-se essencialmente na cadência simétrica a altas BPMs como elemento promotor de vasodilatação.
