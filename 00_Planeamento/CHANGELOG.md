# Changelog: AI Fitness Tracker App Development

Este log rastreia as modificações, melhorias e implementações de funcionalidades da aplicação Fitness Tracker.

## [2026-07-18] Pequenas Correções, Localização da UI & Escrita de Relatório (Refinamentos Gerais)

### Added
*   **Seletor de Idioma (`LandingActivity`)**: Adicionado um botão local no `LandingActivity` (`AppCompatDelegate`) para que novos utilizadores possam ler a Pipeline de registo no idioma preferido de imediato, sem afetar o `UserProfile`.

### Modified
*   **Correções de Código**: Resolução de um pequeno bug de scope `@Composable` no cálculo de `strings` na `DashboardActivity`.
*   **Uniformização do Idioma**: A app e documentação foram totalmente traduzidas/uniformizadas para a língua nativa Portuguesa. Todas as *hardcoded strings* em 8 atividades UI foram extraídas para `strings.xml` e `strings-pt.xml`, preservando os termos técnicos em Inglês.
*   **Relatório Final**: Início imediato da planificação e redação em massa do relatório final de entrega (escrito logo a seguir a esta operação).

## [2026-07-16] Hotfixes de Usabilidade & Localização Global da App (Melhorias Pós-Teste)

### Added
*   **In-App Language Toggle (`UserProfile.kt`, `RegisterActivity`, `EditProfileActivity`)**: Resposta direta ao feedback dos testes de usabilidade do grupo de baixa literacia! Adicionado um seletor manual de idioma (Sistema/Inglês/Português) no registo de conta e edição de perfil (`AppCompatDelegate` via Compose). Implementada migração lazy de dados NoSQL para que utilizadores existentes adotem o idioma do Sistema nativamente sem scripts de base de dados forçados.
*   **Localização Global da App (`res/values-pt/strings.xml`)**: Traduzida toda a UI para Português (pt-PT) para eliminar a barreira linguística em utilizadores de baixa literacia do Grupo A, mantendo termos específicos de fitness (Squats, Lunges, XP, Level) em Inglês.

### Modified
*   **Aumento de Visibilidade HUD (`OverlayView.kt`)**: Redesign dos parâmetros de escala do overlay (contador de repetições 150f, espessura de linha 18f) para garantir legibilidade ideal a 2-6 metros de distância. (Trabalha no Milestone 3)
*   **Arquitetura de Rastreamento Lunge (`LungeExercise.kt`)**: Remodelação do rastreio para seguir o joelho frontal em vez do traseiro, corrigindo permanentemente os falsos negativos causados pela oclusão mecânica da perna em perfil. (Trabalha no Milestone 2)
*   **TTS Debounce (`TTSHelper.kt`)**: Confirmada a implementação de um debounce de 500ms para avisos de áudio de modo a evitar sobreposição de vozes em repetições consecutivas rápidas. (Trabalha no Milestone 3)

## [2026-07-12] Testes de Usabilidade & Registo de Resultados

### Added
*   **Logs de Avaliação de Usabilidade (`resultados_testes_usuabilidade.md`)**: Registadas métricas e raw logs para 7 sessões de testes de usabilidade reais. Os participantes foram divididos em 3 grupos distintos de literacia tecnológica (baixa, média, avançada) e compilados dados críticos sobre a UX/visão computacional.
*   **Pasta para Assinaturas de Consentimento (`Assinaturas_decl_consentimento/`)**: Adicionada sub-pasta e ficheiro de placeholder para arquivar assinaturas digitais/verificações de consentimento dos participantes.

### Modified
*   **Guia de Teste de Utilizador (`Guião  Testes Utilizadores.md`)**: Revisão do fluxo de perguntas, correção da sequência de numeração e clarificação de tarefas para melhor consistência de teste.

## [2026-07-08] Form Scoring Contínuo & Rep-Count Threshold (Melhora Milestone 3)

### Added
*   **Rep-Count Threshold (`ExerciseConfig.kt`)**: Adicionado o `countThresholdDeg`, uma profundidade mais indulgente do que o `idealMinAngleDeg`. Uma repetição agora conta assim que atravessa este limite flexível, em vez de ser ignorada se não chegar à profundidade ideal.
*   **Scoring de Alinhamento na Plank (`PlankExercise.kt`)**: A Plank agora tem um form score real. O alinhamento é pontuado continuamente contra uma linha reta de 180° entre ombro-anca-joelho, caindo de valor quanto mais o corpo ceder ou subir. Cada segmento mantido é gravado no `repHistory` com a sua pontuação.

### Modified
*   **Penalização de Profundidade Contínua (`FormEvaluator.kt`)**: Substituída a verificação de tolerância fixa por uma escala linear contínua: 0 penalidade na profundidade ideal, escalando para -40 exatamente no count threshold.
*   **Lógica de Contagem de Reps (`RepPhaseTracker.kt`)**: O limite inferior do movimento usado para decidir quando uma rep conta agora lê do `countThresholdDeg` em vez de `idealMinAngleDeg`, separando "isto contou como uma rep" de "quão boa foi a rep".
*   **Configurações de Exercícios (`SquatExercise.kt`, `PushUpExercise.kt`, `LungeExercise.kt`)**: Adicionados valores de `countThresholdDeg` - 90° para Squat e Push-up, 107° para Lunge - de forma a que tentativas superficiais contem como repetições mas com menor score.

---

## [2026-07-05] Refinamentos de Usabilidade & Custom Plan Creator (Preparação para Testes)

### Added
*   **Custom Plan Creator (`CreatePlanActivity.kt`)**: Adicionado um ecrã construtor de planos em Compose. Força inserções de passos de descanso ($\ge 30$ segundos) entre todos os exercícios. Valida limites de escala: Squat (5-25 reps), Push-Up (3-15 reps), Lunge (5-20 reps por perna) e Descanso (30-120s). Guarda na Firebase Firestore e passa dinamicamente. (Trabalha no Milestone 1 e 4)
*   **Disclaimer & Calibration Popup (`StartPlanActivity.kt`)**: Adicionado um AlertDialog com isenção de responsabilidade antes de o plano iniciar. Aconselha calibração de câmara de 2 a 6 metros com corpo todo no frame e fornece breves descrições das metas de forma.
*   **Prefixo de Perna Ativa para Lunges (`LungeExercise.kt`)**: Rastrea automaticamente a perna frontal ativa (LEFT ou RIGHT) com base no mapeamento de coordenadas, precedendo-a nos cues do utilizador.
*   **Ícone da Aplicação Estilizado**: Adicionado um ícone de halteres branco inclinado a 45 graus sobre fundo técnico escuro com grelhas, reduzido a 65% para caber em todos os device mask shapes (círculo, lágrima, squircle).

### Modified
*   **Overlay HUD Size Scaling (`OverlayView.kt`)**: Aumento drástico do tamanho das fontes (Title para $65\text{f}$, Big Reps count para $110\text{f}$ e Cues para $64\text{f}$) para visibilidade a 5 metros.
*   **TTS Voice Debounce (`TTSHelper.kt`)**: Implementado debounce post-delayed de 500ms para alertas posturais usando o Android Handler para evitar encavalitamentos. Flush da fila anuncia repetições de imediato. Idioma do motor alterado para Português.
*   **Brief Cues (`FormEvaluator.kt` & Exercises)**: Cues curtos traduzidos e otimizados para Português (e.g. `"Desce!"`, `"Sobe!"`, `"Excelente!"`).
*   **Proteção Leaderboard Wrap (`ViewStatisticsActivity.kt`)**: Reorganização dos items nas linhas empilhando o nome e as estatísticas na vertical para proteger a UI de nomes muito grandes.
*   **Proteção de Mock Seeding (`ViewStatisticsActivity.kt`)**:
    *   Removido botão "Clear History" da UI para proteger a eliminação acidental de dados.
    *   Lote de seed modificado para anexar treinos à lista existente em vez de os substituir.

## [2026-07-04] NoSQL Write-Time Aggregation & Leaderboards (Trabalha no Milestone 1 e 4)

### Added
*   **Write-Time Client Aggregation Fields (`UserProfile.kt`)**: Adicionados campos de stats para manter o desempenho diretamente no document do utilizador:
    *   `totalKcal`, `totalReps`, `totalWorkouts`, `overallCadenceStability` (lifetime).
    *   `weeklyKcal`, `weeklyCadenceStability`, `weeklyWorkouts` (weekly reset).
    *   `lastWeeklyReset` (weekly reset marker).
*   **Leaderboards View (`ViewStatisticsActivity.kt`)**: Adicionada a tab de leaderboards globais mostrando os 10 melhores utilizadores classificados por XP, Kcal ou Cadence Stability.
*   **Workout Detail Dialog (`ViewStatisticsActivity.kt`)**: Clickable cards no histórico abrem agora um popup com análise biomecânica completa: Volume (Reps $\times$ Weight), Cadence Score, Concentric/Eccentric tempos, e Standard Deviation.

### Modified
*   **Transactional Stats Write (`MainActivity.kt`)**: Expandida a transaction de conclusão de treino. Calcula agora a deviation standard da cadence, traduz para um score de 0 a 100, e atomicamente agrega resultados no lifetime e weekly, lidando com auto-reset baseado numa fronteira de semana do calendário.

---

## [2026-07-01] Reestruturação de Entrega & Rascunho LaTeX

### Added
*   **Estrutura de Pastas Raiz (Guias ISEL)**: Criadas pastas standard de projeto: `00_Planeamento`, `01_Analise`, `02_Desenho`, `03_Implementacao`, `04_Teste`, e `_RELATORIO`.
*   **Índices Ficheiros Raiz**:
    *   `_README.TXT`: Descrição do repositório, autores (Rodrigo Correia #45155, David Delgado #51598) e layout.
    *   `prompt_set.TXT`: Log estruturado das AI prompts usadas no core.
*   **Relatório Draft (`_RELATORIO/overleaf/`)**: Criado e rasconhado todo o template LaTeX na pasta `_RELATORIO/overleaf` (abstract, metadata e Chapters 1 a 6).

### Modified
*   **Realocação do Projeto**: Movido `AndroidApp_V_0.1/` e `POC_Python/` para dentro de `03_Implementacao/`.
*   **Caminhos Gitignore**: Configurado `.gitignore` para ignorar `documentos fornecidos projeto/` e `_RELATORIO/overleaf/` prevenindo commits indesejados.

---

## [2026-06-16] Cues de Orientação de Áudio Text-To-Speech (Trabalha no Milestone 3)

### Added
*   **Coaching Assistant (`TTSHelper.kt`)**: Implementado o manager nativo `TextToSpeech` no Android. Rate-limited para prevenir overlapping de voz (cooldown de 4 segundos), mas com override imdediato para conclusões de reps ou contagens decrescentes.
*   **Aviso na UI de Calibração (`OverlayView.kt`)**: Adicionado texto de notificação avisando do uso de orientação audível.

### Modified
*   **Guided Workout Plan Audio Cues (`MainActivity.kt`)**:
    *   O TTS anuncia o início de cada passo do treino ou pausas de descanso.
    *   Anuncia cada repetição completada com o seu Score e Form Coaching notes (e.g. *"Rep 3. Score 88. Desce mais."*).
    *   Avisos em real-time sobre postura.
*   **Single Exercise Testing Audio Feedback (`ExerciseTestActivity.kt`)**:
    *   Conta audivelmente a preparação (e.g., *"Get ready! 5, 4, 3, 2, 1, Go!"*).

---

## [2026-06-15] Integração Workout Plan & Statistics Dashboard (Trabalha no Milestone 4)

### Added
*   **Ecrã Workout Plan Detail (`StartPlanActivity.kt`)**: Apresenta a visão geral de treino. Mostra badges da duração estimada, MET e consumo de Kcal.
*   **Ecrã Activity Statistics (`ViewStatisticsActivity.kt`)**: Implementado o dashboard completo em Jetpack Compose:
    *   **Live Firestore Query**: Traz a info do `/users/{uid}/workouts` ordenada.
    *   **Custom Canvas Bar Chart**: Desenha barras gráficas sobre progresso diário em `Canvas`. Inclui animação e neon sweep gradients.
    *   **Biometric Stats Cards**: Agrega total de workouts e tempo ativo.
    *   **Developer Actions**: Adicionado botão para limpar o histórico e semear Mock Data (5 treinos aleatórios).

### Modified
*   **Guided Workout Plan Loop (`MainActivity.kt` & `WorkoutManager.kt`)**: Expostos steps de treino no `WorkoutManager` para a sequência Squat-Rest-Lunge.
*   **Workout Persistence & Scoring (`MainActivity.kt`)**:
    *   Calcula Kcal MET: `MET (6.0) * 3.5 * weight / 200 * (duration / 60)`.
    *   Guarda treino na sub-collection `/users/{uid}/workouts`.
    *   Update `xpPoints` e `level` em Firestore Transaction: `(reps * 10) + (avgScore / 2)` XP.
    *   Lança `ResultActivity` passando métricas agregadas.

---

## [2026-06-10] Auth, Navegação e Base de Dados (Trabalha no Milestone 1)

### Added
*   **Status de Amigos Online/Offline & Two-way Confirmation Requests**:
    *   `lastActive` timestamp tracking no modelo `UserProfile`.
    *   **Two-Way Friend Request Confirmation Flow**: Pedidos pendentes na sub-collection `friend_requests`. Adicionar amigos cria requests em vez de adicionar instantaneamente.
    *   **Pending Requests UI Section**: Tabela de requests reais no `DashboardActivity.kt` com botões "Accept" e "Decline".
    *   **Transactional Accept Flow**: "Accept" lança Transação atómica que atualiza ambas as friendsLists e elimina o documento de request.
    *   **Real-time Snapshot Syncing**: Refatoração do `DashboardActivity.kt` e `EditProfileActivity.kt` para usar Snapshot Listeners (`addSnapshotListener` no Compose `DisposableEffect`).
*   **Forgot Password Dialog**: Dialog stateful integrado em `LandingActivity.kt`.
*   **Sign Out Action**: Botão vermelho em `DashboardActivity.kt`.
*   **First Name Parser**: O dashboard mostra unicamente o primeiro nome (e.g. "Welcome Back, Rodrigo").
*   **Loader Activity**: Criada `LoaderActivity.kt` com animado neon spinner.
*   **Tablet Layout & Centering Constraints**: Limitada largura do ecrã para evitar estiramento horizontal exagerado nos tablets (`Modifier.widthIn(max = 480.dp)`).
*   **Locked Portrait Mode**: `AndroidManifest.xml` bloqueado em portrait mode.

### Modified
*   **Prevent Screen Locking**: Mantém o ecrã ativo na app via `FLAG_KEEP_SCREEN_ON`.
*   **Human-Readable Firestore Dates**: O sistema agora usa e analisa objetos reais `Timestamp` do Firestore em vez do long timestamp para melhor compatibilidade.
*   **User Numeric ID**: Códigos aleatórios de 5 dígitos na criação da conta (e.g. `#12345`).
*   **Transitions via Loader**: Fluxo da `LandingActivity` e `DashboardActivity` para o treino usa o Ecrã de `LoaderActivity`.

---

## Histórico Inicial do Projeto (Fundamentos e Análise)

### [2026-06-09]
*   Planeamento da estrutura de navegação.
*   Implementação Navegação entre as Activities da aplicação (exploração do fluxo de ecrãs)
*   Definição e configuração de temas visuais para a interface
*   Planeamento de métodos autenticação de utilizadores e Integração na App.

### [2026-05-18]
*   Preparação da apresentação para o FEIM 2026.
*   Avaliação do Ponto de Situação do projeto e Redefinição de Prioridades.
*   Pausa curta no Projeto para fechar outras unidades curriculares.

### [2026-05-07]
*   Desenho dos layouts da aplicação no Figma.
*   Coletânea de documentos (Google Scholar) para criação de métricas / estudo do problema / lógica do negócio.
*   Identificação de use case para Ollama: Criação de VM na cloud para integração via API sobre as features construídas no projeto (ex: Feedback via ollama do regime de treinos e resultados).
*   Enquadramento com Orientador de Projeto sobre casos críticos a resolver até ao FEIM.

### [2026-04-25]
*   Adaptação do código python para um projeto Android, dando origem ao esqueleto de desenvolvimento do produto final.
*   Testes realizados e avaliação do estado atual de desenvolvimento. Planeamento em conjunto com o orientador.

### [2026-04-19]
*   Após vários estudos e abordagens ao longo da semana, foi decidido que a melhor abordagem para o desenvolvimento do projeto será a implementação do motor em conjunto com a app em Kotlin. Deste modo, o desenvolvimento das funcionalidades destinadas à app ficam inseridas no ambiente Android que facilita a ponte com o front-end da aplicação final.

### [2026-03-24]
*   Realização dos casos de utilização e requisitos funcionais e atributos do sistema. Documentos disponíveis na drive.

### [2026-03-17]
*   Criação do repositório.
*   Criadas pastas e classes para iniciar o desenvolvimento de um simples projeto piloto em Python. Desenvolvimento de uma estrutura base para planeamento da aplicação.
