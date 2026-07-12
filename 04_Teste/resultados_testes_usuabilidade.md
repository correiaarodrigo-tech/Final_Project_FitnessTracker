# Resultados dos Testes de Usabilidade — Fitness Tracker

Este documento regista as métricas, observações e análises extraídas das sessões de testes de usabilidade realizadas com utilizadores reais. O objetivo principal consistiu em validar a autonomia dos utilizadores na interface e a eficácia do algoritmo de visão por computador baseado em ângulos (BlazePose MediaPipe).

---

## 1. Quadro de Segmentação por Perfis (Personas)

Para garantir a representatividade dos testes, os 7 participantes foram mapeados em 3 grupos distintos de acordo com a sua literacia tecnológica e facilidade de interação com sistemas digitais. Para esta segmentação consideramos apenas conhecimentos relativos ao uso de dispositivos móveis e não conhecimentos da área da 'motricidade e fitness'.

| Grupo de Literacia | Participante | Idade | Foco Principal do Teste |
| :--- | :--- | :---: | :--- |
| **Grupo A: Literacia Baixa**<br>*(Apenas conhecimentos simples de uso de telemóvel, geralmente necessitam apoio para usos menos convencionais fora de chamadas e mensagens)* | **Leandra**<br>**António** | 28<br>26 | Validação Modelo Avaliação exercício, dependência de assistência para iniciar e desenho aplicação. |
| **Grupo B: Literacia Média**<br>*(Utilizadores comuns de smartphone, utilização banal de smartphones e outras tarefas simples como conectar telemóvel a um dispositivo bluetooth)* | **Mónica**<br>**Carlos**<br>**Mafalda** | 55<br>58<br>27 | Validação Modelo Avaliação exercício, dependência de assistência para iniciar. |
| **Grupo C: Literacia Avançada**<br>*(Área profissional tecnológica, ou similar. Conhecimentos de programação ou similar. Utilizador frequente de aplicações variadas para diversas necessidades.)* | **Tomás**<br>**Tiago** | 22<br>25 | Validação Modelo Avaliação exercício, dependência de assistência para iniciar e desenho aplicação. Limites da avaliação / classificação. |

---

## 2. Diário de Bordo e Anotações das Sessões (Recolha Direta)

Anotações brutas registadas em tempo real pelo observador durante a execução dos testes.

### Sessão 01: Leandra
* **Ficha de Metadados:** Data: 04/07/2026 | Dispositivo: Próprio | Distância: 2.8m | Iluminação: Boa/Direta | Outros: Sala ampla, calças contrastantes.
* **Fase 0 (Autonomia):** Login hesitante devido aos termos em inglês; precisou de 1 pista para avançar no menu. Botão de início encontrado após exploração rápida.
* **Fase 1 (Protocolo & Ângulos):** Deteção inicial demorada (corpo de perfil). Squats: Correta. Lunges: Correta. Push-ups: Falhou 1 contagem. Oclusão notada no chão.
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Sentiu hesitação com o inglês no login.
  * *P2 (Ritmo/Amplitude):* Adaptou inicialmente; depois correu fluido.
  * *P3 (Posicionamento/Luz):* Sem problemas de luz; distância fácil.
  * *P4 (Falhas específicas):* Notou atraso no feedback das flexões.

### Sessão 02: António
* **Ficha de Metadados:** Data: 04/07/2026 | Dispositivo: Observador | Distância: 2.5m | Iluminação: Fraca/Penumbra | Outros: Quarto com espaço reduzido.
* **Fase 0 (Autonomia):** Errou a password duas vezes; bloqueou no jargão técnico do menu inicial. Precisou de ajuda direta para iniciar o treino.
* **Fase 1 (Protocolo & Ângulos):** Deteção inicial falhou (exigiu reposicionar candeeiro). Squats: Falsos positivos (contou 6). Lunges: Falhou contagem. Push-ups: Falhou contagem.
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Fluxo inicial confuso e muito técnico.
  * *P2 (Ritmo/Amplitude):* Teve de exagerar a descida nos squats para contar.
  * *P3 (Posicionamento/Luz):* Muito difícil enquadrar num quarto pequeno.
  * *P4 (Falhas específicas):* Lunges falharam de forma sistemática.

### Sessão 03: Mónica
* **Ficha de Metadados:** Data: 11/07/2026 | Dispositivo: Observador | Distância: 3.0m | Iluminação: Contra-luz (Janela de fundo) | Outros: Roupa escura.
* **Fase 0 (Autonomia):** Login fluido (reconheceu os campos básicos). Hesitou no menu ao procurar os exercícios livres.
* **Fase 1 (Protocolo & Ângulos):** Deteção inicial demorada devido ao forte contraste da janela. Squats: Correta. Lunges: Correta. Push-ups: Falhou contagem (perda de tracking no chão).
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Entrou bem, mas o menu podia ser mais simples.
  * *P2 (Ritmo/Amplitude):* Movimentos normais, o algoritmo acompanhou bem de pé.
  * *P3 (Posicionamento/Luz):* Janela atrás causou problemas no início.
  * *P4 (Falhas específicas):* Não contou nenhuma flexão no chão.

### Sessão 04: Carlos
* **Ficha de Metadados:** Data: 12/07/2026 | Dispositivo: Observador | Distância: 3.2m | Iluminação: Boa/Direta | Outros: Espaço de garagem.
* **Fase 0 (Autonomia):** Avançou sem apoio no login. Encontrou o botão de treino rapidamente através dos ícones visuais.
* **Fase 1 (Protocolo & Ângulos):** Deteção inicial imediata. Squats: Falhou contagem (2 não registadas). Lunges: Correta. Push-ups: Correta.
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Fluxo muito direto e sem erros.
  * *P2 (Ritmo/Amplitude):* Fez movimento mais curto e a app não contou.
  * *P3 (Posicionamento/Luz):* Fácil de posicionar na garagem.
  * *P4 (Falhas específicas):* Squats falharam por falta de amplitude profunda.

### Sessão 05: Mafalda
* **Ficha de Metadados:** Data: 12/07/2026 | Dispositivo: Próprio | Distância: 2.7m | Iluminação: Boa/Direta | Outros: Praticante de Pilates.
* **Fase 0 (Autonomia):** Registo autónomo fluido. Encontrou o ecrã de treino de imediato e iniciou sem qualquer instrução.
* **Fase 1 (Protocolo & Ângulos):** Deteção imediata. Squats: Correta. Lunges: Correta. Push-ups: Correta (execução lenta ajudou o tracking).
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Sem qualquer confusão, interface limpa.
  * *P2 (Ritmo/Amplitude):* Execução controlada funcionou na perfeição.
  * *P3 (Posicionamento/Luz):* Enquadramento fácil à primeira.
  * *P4 (Falhas específicas):* Nada a apontar, contagem exata.

### Sessão 06: Tomás
* **Ficha de Metadados:** Data: 12/07/2026 | Dispositivo: Próprio | Distância: 2.8m | Iluminação: Boa/Direta | Outros: Ritmo de execução elevado.
* **Fase 0 (Autonomia):** Fluxo imediato. Navegação intuitiva e instantânea pelas opções.
* **Fase 1 (Protocolo & Ângulos):** Deteção imediata. Squats: Correta. Lunges: Correta. Push-ups: Falsos positivos (contou repetições fantasma na subida rápida).
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Super simples, padrão comum de apps de treino.
  * *P2 (Ritmo/Amplitude):* Execução rápida gerou pequenos bugs visuais no esqueleto.
  * *P3 (Posicionamento/Luz):* Sem problemas encontrados.
  * *P4 (Falhas específicas):* Flexões contaram repetições a mais devido à velocidade.

### Sessão 07: Tiago
* **Ficha de Metadados:** Data: 12/07/2026 | Dispositivo: Observador | Distância: 4.1m | Iluminação: Boa/Direta | Outros: Utilizador muito alto (>1,90m), exigiu recuar muito o dispositivo.
* **Fase 0 (Autonomia):** Login rápido. Encontrou o botão de treino sem dificuldades.
* **Fase 1 (Protocolo & Ângulos):** Deteção demorada (dificuldade em enquadrar pés e cabeça em simultâneo). Squats: Falsos positivos. Lunges: Correta. Push-ups: Falhou contagem.
* **Fase 2 (Entrevista Qualitativa):**
  * *P1 (Confusão inicial):* Fluxo inicial simples e direto.
  * *P2 (Ritmo/Amplitude):* Amplitude normal, mas deteção instável pela distância.
  * *P3 (Posicionamento/Luz):* Muito difícil posicionar a câmara para cobrir a altura total.
  * *P4 (Falhas específicas):* Falhas frequentes no chão (Push-ups).

---

## 3. Matriz de Respostas ao Questionário Completo (SUS + Métricas Descritivas)

Classificações atribuídas numa escala de 1 (**Discordo Totalmente**) a 5 (**Concordo Totalmente**).



---

## 4. Observações Gerais dos Resultados (Análise Consolidada)

### Interface e Usabilidade Geral (Fases 0 e 2)
* **Ausência de Localização (Idiomas):** A falta de um seletor PT/EN gerou entropia imediata no Grupo A (Leandra e António), causando hesitações no ecrã de registo.
* **Ritmo do Feedback por Áudio (*Text-to-Speech*):** O motor de voz revelou-se demasiado rápido. Em momentos de fadiga física ou acumulação rápida de repetições, as frases sobrepunham-se ("atropelavam-se"), penalizando a clareza do feedback (P12) nos Grupos A e B.

### Comportamento do Algoritmo e Ambiente (Fases 1 e 3)
* **Problema de Escala e Distância (Acessibilidade Visual):** À distância necessária para o BlazePose capturar o corpo inteiro (2.5m a 4.1m), o texto do ecrã ficou virtualmente ilegível para a maioria. Utilizadores quebraram a postura correta para tentar ler o progresso no visor.
* **Dificuldade de Posicionamento Autónomo (P15 e P16):** Utilizadores com maior estatura (Tiago) ou quartos com menos espaço (António e Mãe) sentiram grande complexidade em calibrar o ângulo ideal sem recorrer à ajuda de um terceiro.
* **Fidelidade da Contagem por Exercício:** 
  * **Squats e Lunges:** Excelente consistência em execuções controladas. Registaram-se falhas (falsos negativos) em amplitudes curtas (Pai) e falsos positivos em execuções erráticas.
  * **Push-ups:** Apresentou a maior taxa de erro de tracking devido à oclusão dos nós articulares dos cotovelos/ombros no plano horizontal do chão.
