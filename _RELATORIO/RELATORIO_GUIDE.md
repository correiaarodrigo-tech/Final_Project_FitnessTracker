# Guia de Estrutura do Relatório (Opção Alternativa — Máx. 20 Páginas)

Este documento serve como mapa de planeamento para redigir o relatório oficial do projeto de fim de curso em LaTeX, no diretório `_RELATORIO/overleaf/`.

---

## 📘 PARTE 1 — Decisões e Desenho (Máx 10 Páginas)

### Capítulo 1: Introdução e Motivação (~2 Páginas)
*   **Enquadramento do Problema**: Prática de exercício físico doméstico sem acompanhamento profissional; elevado risco de lesões por postura incorreta e falta de motivação (gamificação).
*   **Objetivos do Projeto**: Desenvolver uma aplicação Android nativa que use inteligência artificial para monitorizar a postura do utilizador em tempo real, calcular métricas biomecânicas e motivá-lo com um sistema competitivo.
*   **Decisão de Processamento Local (Edge AI)**: Justificar o uso de MediaPipe BlazePose rodando diretamente no telemóvel para menor latência, privacidade e ausência de custos com servidores de processamento de vídeo.
*   **Contribuições do Trabalho**:
    *   Modelo matemático para cálculo de ângulos e máquina de estados de repetição.
    *   Algoritmo de avaliação de postura baseado em referências oficiais de cinesiologia (ACSM, NSCA).
    *   Arquitetura de dados NoSQL pre-agregada na escrita (Write-Time Aggregation) para leaderboards escaláveis.

### Capítulo 2: Decisões e Desenho Técnico (~4 Páginas)
*   **Arquitetura do Cliente**: Uso de padrão MVVM (Model-View-ViewModel), Jetpack Compose para UI declarativa e CameraX para pipeline fluido de vídeo.
*   **Arquitetura do Servidor (NoSQL Firestore)**:
    *   **Estratégia de Agregação na Escrita (Write-Time Aggregation)**: Detalhar a transação no código Kotlin que atualiza os recordes do utilizador no momento do commit do treino.
    *   **Análise de Prós e Contras**:
        *   *Prós*: Leituras na base de dados altamente otimizadas ($O(1)$) para renderizar gráficos semanais e leaderboards globais. Sem custos de leitura excessivos.
        *   *Contras*: Complexidade extra no código Kotlin da transação.
*   **Métricas Biomecânicas e Cinemáticas (Limiares)**:
    *   Apresentar os limiares de ângulo de flexão e extensão das articulações para os exercícios implementados (Agachamento $\le 70^\circ$, Flexão $\le 70^\circ$, Afundo $\le 80^\circ$).
    *   Citar a fundamentação científica: ACSM (cadências e teste de flexão), NSCA (alinhamento espinhal e agachamento paralelo) e FITescola (protocolo de falha e ritmo).

### Capítulo 3: Usabilidade e Validação Prática (~3 Páginas)
*   **Protocolo de Teste de Utilizadores**: Resumo da aplicação do guião de testes localizado em `04_Teste/Guião  Testes Utilizadores.md` a um grupo de 5-10 pessoas.
*   **Resultados de Usabilidade (SUS)**: Apresentação da pontuação média no formulário SUS e análise de satisfação.
*   **Feedback Qualitativo e Melhorias**: Discussão das observações dos testes (ex: legibilidade do ecrã a 2 metros, clareza dos alertas de áudio TTS, velocidade do detector de calibração).

### Capítulo 4: Conclusão e Trabalho Futuro (~1 Página)
*   **Conclusão**: Síntese dos objetivos atingidos (deteção de pose local a 30 FPS, feedback por áudio funcional, tabelas de classificação).
*   **Trabalho Futuro**: Sugestões de melhoria (ex: suporte a tripés, ajuste dinâmico de volume de áudio, feedback vibratório).

---

## 📑 PARTE 2 — Evidências e Rastreabilidade (Máx 10 Páginas)

### Capítulo 5: Engenharia de Prompts e Rastreabilidade de IA (~3 Páginas)
*   **Conjunto de Prompts Utilizado**: Resumo e seleção dos prompts críticos de `prompt_set.TXT` que foram usados para gerar as partes complexas do software (ex: renderização do Compose Canvas, máquina de estados do tracker de pose e transações Firestore).
*   **Processo de Pair-Programming**: Descrição de como o grupo refinou as sugestões de código da IA para garantir segurança de tipos e tratamento de nulos no ecossistema Android.

### Capítulo 6: Implementação de Código Crítico (~4 Páginas)
*   **Máquina de Estados de Repetição**: Mostrar trechos de código do `RepPhaseTracker.kt` demonstrando a lógica de transição entre estados (`AT_TOP` -> `DESCENDING` -> `DESCENDED` -> `ASCENDING` -> `AT_TOP`).
*   **Pontuação Postural e Ritmo**: Exposição do código em `FormEvaluator.kt` que calcula deduções matemáticas de ROM e cadência de descida/subida.
*   **Transação Firestore**: Exposição do trecho de código em `MainActivity.kt` que atualiza atomicamente os agregados de Kcal, XP, Volume e Cadência na base de dados.

### Capítulo 7: Evidências Visuais e de Execução (~3 Páginas)
*   **Interface Gráfica**: Capturas de ecrã do dashboard de estatísticas, gráfico Canvas neon e o pop-up de detalhe do treino.
*   **Competição (Leaderboards)**: Capturas de ecrã da aba de leaderboards populada em tempo real a partir de dados agregados de vários utilizadores.
*   **Testes de Compilação**: Logs de sucesso da compilação Gradle no terminal, provando a qualidade estática do código Kotlin.
