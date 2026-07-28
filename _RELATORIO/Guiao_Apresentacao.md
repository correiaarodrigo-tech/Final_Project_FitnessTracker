# Guião de Apresentação — FitnessTracking (PRJ #64)

**Duração alvo: ~10 minutos.** Os tempos por slide são indicativos — ensaia com cronómetro e ajusta. Total estimado: ~10m00s — já sem margem, ensaia bem para não passares do tempo.

---

### Slide 1 — Título (~10s)

> "Boa tarde. Somos o Rodrigo Correia e o David Delgado, e vamos apresentar o FitnessTracking: uma aplicação Android que usa a câmara do telemóvel para detetar exercícios físicos em tempo real."

---

### Slide 2 — Índice (~15s)

> "A apresentação está dividida em cinco partes: o problema que motivou o projeto, a arquitetura técnica, o rigor científico e as decisões que fomos adaptando, a validação com utilizadores reais, e por fim o produto final."

---

### Slide 3 — Capítulo 01 · O Problema (~5s)

> "Comecemos pelo problema."

---

### Slide 4 — O Problema (~45s)

> "O treino em casa cresceu muito pela conveniência, mas trouxe um risco real: sem um profissional a observar, erros de postura e de cadência passam despercebidos. E a monitorização manual — olhar para o telemóvel, contar em voz alta — distrai do próprio exercício.
>
> O nosso objetivo foi construir uma app Android nativa que usa só a câmara frontal do telemóvel para detetar poses, contar repetições de forma autónoma, e corrigir a execução em tempo real."

*(Aponta para o placeholder de imagem à direita, se já tiveres uma captura de ecrã da app inserida.)*

---

### Slide 5 — Requisitos (~35s)

> "Isto traduziu-se em três exigências que moldaram toda a solução: primeiro, sem sensores dedicados — só a câmara do telemóvel, sem wearables nem hardware extra. Segundo, feedback imediato — correções visuais e sonoras durante o próprio exercício. E terceiro, teria de funcionar no dispositivo — processamento local, sem depender de servidores."

---

### Slide 6 — Capítulo 02 · Arquitetura (~5s)

> "Vamos ver como o sistema funciona."

---

### Slide 7 — Arquitetura (~45s)

> "A arquitetura segue um pipeline simples: a câmara, via CameraX, captura o vídeo; o MediaPipe BlazePose deteta a pose corporal; um motor cinemático em Kotlin analisa os ângulos articulares; e os resultados chegam à interface, feita em Jetpack Compose, com sincronização opcional no Firebase.
>
> A deteção é feita em duas fases, seguindo Bazarevsky et al. (2020): um detetor localiza o corpo apenas na primeira frame, e um modelo de rastreio mais leve segue os landmarks nas seguintes, sem repetir a deteção completa.
>
> Um ponto importante: o vídeo nunca sai do telemóvel. É Edge Computing puro — sem servidores, sem latência de rede, e mais privacidade para o utilizador."

---

### Slide 8 — Máquina de Estados (~40s)

> "O motor de avaliação funciona como uma máquina de estados com três fases: AT_TOP, quando o utilizador está em repouso à espera do início do movimento; DESCENDING, a fase excêntrica, a descer; e ASCENDING, a fase concêntrica, a subir, que conclui a repetição.
>
> Usamos três limiares diferentes: um limiar de extensão que inicia e fecha a repetição, um limiar de contagem mais permissivo que decide se ela é válida, e um limiar ideal que decide a pontuação. Isto evita falsos positivos causados por tremeluzir da câmara."

---

### Slide 9 — Limiares (~40s)

> "Estes são os limiares reais para os quatro exercícios implementados — Agachamento, Flexão de Braços, Lunge e Prancha — cada um com o seu limiar ideal, de contagem, e de extensão.
>
> Um exemplo prático: um agachamento a 65 graus de profundidade vale 100 pontos. O mesmo agachamento, mas mais raso, a 82 graus, vale só 76 pontos — a penalização de amplitude de movimento é proporcional, não é tudo ou nada."

---

### Slide 10 — Como se Calcula a Pontuação (~40s)

> "Cada repetição começa com 100 pontos, e perde pontos por duas razões distintas. A primeira é a profundidade, de forma proporcional: o erro é a diferença entre o ângulo atingido e o ideal, e a penalização é esse erro a dividir pelo intervalo até ao limiar de contagem, vezes 40 pontos no máximo. A segunda é a cadência — uma dedução fixa, não proporcional, se a fase excêntrica ou concêntrica forem demasiado rápidas ou demasiado lentas.
>
> No nosso Agachamento, com ideal a 70 graus e limiar de contagem a 90: a 65 graus, o erro é zero, e a repetição vale 100 pontos. A 82 graus, o erro é 12, o intervalo é 20, a penalização é 12 sobre 20 vezes 40 — ou seja, 24 pontos — e a repetição vale 76. Confirmámos que estes números batem exatamente certo com o código real."

---

### Slide 11 — Capítulo 03 · Rigor e Adaptação (~5s)

> "Passamos agora ao rigor científico por trás destes números, e às decisões que fomos obrigados a rever."

---

### Slide 12 — Fundamentação Científica (~40s)

> "Os limiares não foram definidos a olho. Apoiámo-nos em três fontes: Bazarevsky et al. (2020) para a arquitetura do BlazePose; Norkin & White, cuja obra de goniometria — através de Rowe et al. — define 110 graus de flexão mínima do joelho para atividades funcionais, o que confirma os nossos 70 e 90 graus do Agachamento sem qualquer ajuste; e Wilk, Zajac & Tufano (2021), que mostram que a fase excêntrica tem influência mais consistente na força e na hipertrofia — o que justifica penalizarmos a descida duas vezes mais do que a subida.
>
> Onde a literatura não dava resposta — como no Lunge e na Prancha — os limiares foram calibrados empiricamente, e depois ajustados com dados reais de utilizadores."

---

### Slide 13 — A História do Lunge (~45s)

> "E isso leva-nos à história do Lunge, o exercício mais difícil de acertar. Passou por três iterações: começámos com um limiar de 80 graus, definido de forma empírica. Depois, por observação visual, subimos para 87 graus, porque estava a rejeitar repetições bem executadas. Mas só nos testes de usabilidade com utilizadores reais percebemos a verdadeira causa: o joelho traseiro ficava oculto pela perna da frente.
>
> A correção não foi tentar detetar a oclusão — foi eliminar o problema na origem: o sistema passou a identificar, a cada frame, qual a perna que está à frente, e a medir só essa."

---

### Slide 14 — Decisões de Âmbito (~35s)

> "Duas decisões mudaram o rumo do projeto. A 19 de abril de 2026, decidimos abandonar o protótipo Python como sistema à parte, e reescrever o motor de deteção nativamente em Kotlin, integrado diretamente na app Android — o protótipo Python ficou como prova de conceito histórica.
>
> A 7 de maio, identificámos uma possível funcionalidade — um assistente de IA local via Ollama para dar feedback motivacional — mas decidimos não a implementar neste semestre, por não constar dos requisitos funcionais. Ficou documentada como trabalho futuro.
>
> Isto mostra capacidade de gerir âmbito: adotar uma arquitetura mais robusta a meio do projeto, e saber dizer não a funcionalidades que não eram essenciais."

---

### Slide 15 — Capítulo 04 · Validação (~5s)

> "Mas nada disto interessa se não funcionar para utilizadores reais. Vamos à validação."

---

### Slide 16 — Validação (~45s)

> "Testámos a app com 7 utilizadores reais, distribuídos por três perfis de literacia tecnológica — baixa, média e avançada. O resultado no System Usability Scale foi 66.42, ligeiramente abaixo da média da indústria, que é 68, com um desvio-padrão de 19.62.
>
> Os testes mudaram coisas concretas: localizámos toda a interface para português, introduzimos um debounce de 0.5 segundos no áudio para evitar sobreposição de instruções, ampliámos as fontes do HUD para leitura a 2.5 a 4 metros de distância, e corrigimos a deteção do Lunge, como já vimos."

---

### Slide 17 — Capítulo 05 · Produto Final (~5s)

> "Por fim, o produto completo."

---

### Slide 18 — Demonstração em Vídeo (~1min)

*(Narra por cima do vídeo em vez de ler o guião — usa isto só como referência do que mostrar.)*

> "Aqui vemos a app a funcionar de ponta a ponta: a deteção da pose e a pontuação em tempo real, seguidas do dashboard de estatísticas, da gamificação com pontos de experiência e nível, e da rede de amigos."

---

### Slide 19 — Produto Final (~35s)

> "O produto final inclui gamificação, com pontos de experiência e nível que refletem a qualidade técnica da execução; leaderboards, com ranking global por XP, calorias, ou estabilidade de cadência; uma rede de amigos, com pedidos de amizade e deteção de estado online; e um dashboard de estatísticas, com gráficos de progresso desenhados em Jetpack Compose Canvas, sem qualquer biblioteca externa."

---

### Slide 20 — Conclusões e Trabalho Futuro (~35s)

> "Em resumo, entregámos quatro exercícios avaliados em tempo real com scoring biomecânico, um motor cem por cento local, gamificação e rede social completas, e validação prática com utilizadores reais.
>
> Como trabalho futuro, fica testes A/B às novas mecânicas de gamificação, desafios competitivos em tempo real entre amigos, testes mais intensivos à oclusão do Lunge noutros contextos, e novos exercícios de membros superiores com avaliação 3D completa."

---

### Slide 21 — Obrigado (~10s)

> "Obrigado pela atenção. Estamos disponíveis para perguntas."

---