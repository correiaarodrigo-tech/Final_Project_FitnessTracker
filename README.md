# Fitness Tracking App

Projeto de final de curso - Aplicação para monitorização de exercícios em tempo real.

## Autores

- [45155 - Rodrigo Correia]
- [51598 - David Delgado]

## Links Úteis

- Drive de recursos -> https://iselpt-my.sharepoint.com/:f:/g/personal/a51598_alunos_isel_pt/IgDmtY9-GcMRT6fJk1uhffsZAcqzwMO2dog03hfgFe_jKHI?e=OK5wBT

## Pré-requisitos iniciais

Estes requisitos dizem respeito ao código desenvolvido até ao momento para um ambiente de testes inicial.

- Python 3.10
- Pip installer
- Bibliotecas referenciadas no ficheiro 'requirements.txt'

## Diário de Bordo


<details>
  <summary><b>17/03/2026</b></summary>

- Criação do repositório.
- Criadas pastas e classes para iniciar o desenvolvimento de um simples projeto piloto em python. Desenvolvimento de uma estrutura base para planeamento da aplicação.

</details>

<details>
  <summary><b>24/03/2026</b></summary>
  
- Realização dos casos de utilização e requisitos funcionais e atributos do sistema. Documentos disponíveis na drive.

</details>

<details>
  <summary><b>19/04/2026</b></summary>
  
- Após vários estudos e abordagens ao longo da semana, foi decidido que a melhor abordagem para o desenvolvimento do projeto será a implementação do motor em conjunto com a app em Kotlin. Deste modo, o desenvolvimento das funcionalidades destinadas a app ficam inseridas no ambiente Android que facilita a ponte com o front end do aplicativo final.

</details>

<details>
  <summary><b>25/04/2026</b></summary>
  
- Adaptação do código python para um projeto Android, dando origem ao esqueleto de desenvolvimento do produto final.
- Testes realizados e avaliação do estado atual de desenvolvimento. Planeamento em conjunto com o orientador.

</details>

<details>
  <summary><b>07/05/2026</b></summary>
  
- Desenho dos layouts da aplicação no figma.
- Coletânea de documentos (Google Schoolar) para criação de métricas / estudo do problema / lógica dp negócio.
- Identificação de use case para Ollama: Criação de VM na cloud para integração via API sobre as features construídas no projeto. Ex: Feedback via ollama do regime de treinos e resultados...
- Enquadramento com Orientador Projeto sobre casos críticos a resolver até ao FEIM.

</details>

<details>
  <summary><b>18/05/2026</b></summary>
  
- Preparação da apresentação para o FEIM 2026.
- Avaliação do Ponto de Situação do projeto. Redefinição Prioridades.
- Pausa curta no Projeto para fechar outras unidades curriculares.

</details>

</details>

<details>
  <summary><b>09/06/2026</b></summary>
  
- Planeamento da estrutura de navegação.
- Implementação Navegação entre as Activities da aplicação (exploração do fluxo de ecrãs)
- Definição e configuração de temas visuais para a interface
- Planeamento de métodos autenticação de utilizadores. Planeamento Integração na App.

</details>

<details>
  <summary><b>15/06/2026</b></summary>
  
- Criação de Vídeos ilustrativos das implementações funcionais da aplicação.
- Implementação suporte audio para exercios. (Em desenvolvimento)
- Planeamento de elementos Gamificação aplicação.
- Planeamento testes com utilizadores.

</details>

<details>
  <summary><b>01/07/2026</b></summary>
  
- Reestruturação completa do repositório para entrega final segundo as normas do ISEL (pastas `00_Planeamento`, `01_Analise`, `02_Desenho`, `03_Implementacao`, `04_Teste` e `_RELATORIO`).
- Relocalização do código da aplicação Android nativa e do protótipo Python para dentro da pasta `03_Implementacao`.
- Configuração do `.gitignore` para ignorar rascunhos LaTeX locais e ficheiros de templates.
- Criação dos ficheiros de índice `_README.TXT` e `prompt_set.TXT` na raiz.
- Elaboração completa do rascunho de todos os capítulos do relatório LaTeX na pasta `_RELATORIO/overleaf/`.
- Verificação e compilação bem-sucedida do código Kotlin da aplicação Android na sua nova localização.

</details>

<details>
  <summary><b>04/07/2026</b></summary>
  
- Implementação da arquitetura NoSQL Firestore com agregação na escrita (Write-Time Aggregation) executada via transação de cliente Kotlin.
- Implementação de ecrãs para listagem de leaderboards (ladders de XP, calorias e cadência) e detalhes de treinos com métricas biomecânicas avançadas (desvio padrão e tempos concêntrico/excêntrico).

</details>

<details>
  <summary><b>05/07/2026 - 06/07/2026</b></summary>
  
- Implementação do criador de planos de treino interativo (Custom Plan Creator) em Compose, com validação de limites fisiológicos e inserção automática de descanso de pelo menos 30 segundos.
- Escalonamento das fontes do HUD Overlay (número de repetições e cues) para leitura clara a mais de 5 metros de distância.
- Otimizações de áudio no motor TTS (Text-to-Speech) com debounce de 0.5s para evitar sobreposições e mudança de idioma para português nativo.
- Integração de feedback de perna ativa (esquerda/direita) nos afundos (lunges) e aviso de distância de calibração (2 a 6 metros) antes do treino.
- Redesenho do ícone da aplicação com um halter diagonal branco sobre fundo técnico preto e grelha minimalista.
- Criação do modelo visual da base de dados Firestore em `02_Desenho/BaseDados_Model.md` e relocalização de `PROJECT_ROADMAP.md` para a pasta `00_Planeamento`.

</details>

<details>
  <summary><b>08/07/2026</b></summary>
  
- Implementação de pontuação contínua por profundidade de repetição em `FormEvaluator.kt`.
- Adição do parâmetro `countThresholdDeg` em `ExerciseConfig.kt` e `RepPhaseTracker.kt` para contar repetições menos profundas penalizando a pontuação em vez de descartar a repetição.
- Reestruturação do `PlankExercise.kt` para registar cada segmento de prancha em `repHistory` com cálculo de pontuação em tempo real com base no alinhamento espinhal de 180°.

</details>

<details>
  <summary><b>12/07/2026</b></summary>
  
- Conclusão e registo de 7 sessões de testes de usabilidade com utilizadores reais divididos em 3 perfis de literacia digital em `04_Teste/resultados_testes_usuabilidade.md`.
- Revisão do guião de testes (`Guião  Testes Utilizadores.md`) e criação de estrutura para arquivo de declarações de consentimento assinado.

</details>


<details>
  <summary><b>16/07/2026</b></summary>
  
  *   **Testes de Usabilidade e Ações Corretivas**: Após a análise dos testes práticos aos 3 Grupos de Literacia Tecnológica, implementámos as seguintes soluções críticas de usabilidade:
     - **Localização Global (pt-PT) e Seletor de Idioma**: Com base direta no feedback dos utilizadores (especialmente o Grupo A com menor literacia), traduzimos integralmente a interface para Português. Adicionámos também um seletor manual de idioma (Sistema/Inglês/Português) na criação de conta e edição de perfil, facilitando o onboarding de novos utilizadores.
     - **Visibilidade à Distância**: Refatoração do HUD (escala e espessura) para visualização clara a 4-6 metros.
     - **Fix Oclusão Lunge**: Alteração do joelho rastreado para a perna frontal, eliminando falsos negativos no Lunge.
</details>

## Objetivos Semanais (14-20 Junho)

- Planeamento de otimizações e métricas sobre lógica negócio (resolução imagem maior ou menor, angulos de exercicio maior ou menor... )
- [x] Realização de testes / estudo MediaPipe. Entender a Lógica de Negócio. Colocar em papel. (Fundamentar o projeto: ' O que é um exercício?' 'Como identificar um exercício?' 'Como o MediaPipe Analisa uma Imagem?'... etc)
- [ ] Desenhar plano de implementação Ollama no projeto.
- [x] Pensar / Desenhar Estrutura de Dados para Utilizadores / Exercícios.
- [x] Reestruturar e organizar Github. Melhorar documentação via Gifs ilustrativos do desenvolvimento entre outros, documentação de procedimentos ou falhas.
- Dica ao construir os slides: 1 min por folha, fazer video da demo em vez de live.
- Outra dica: perguntar a Joana / Jaison problemas identificados com processamento imagem, desafios, soluções... (Projetos idênticos)

## Para pensar...

- modelar classificadores e avalidores mais precisos / eficazes?
- estatisticas e feedback melhorado -> timers,contagens, representacao visual do utilizador?
- aprimorar mais exercicios e series? Parametros? Eficácia??

## Figma

Link - https://www.figma.com/design/wSXHzneXyKGZCTNtxxvbXm/Projeto-Final---FitnessTracking?node-id=0-1&t=3xgmSqZSZ6habhqy-1

## Funcionalidades Aplicação

###  Autenticação e Login
https://github.com/user-attachments/assets/9a9a8690-698d-446a-a72c-90059b276c34
###  Dados Utilizador e edição
https://github.com/user-attachments/assets/226ddb1f-8874-43c8-9075-0b979594e4b8
###  Estatisticas Utilizador
https://github.com/user-attachments/assets/bd1314a2-cb20-4b94-8589-b98f078e5fce
###  Vista geral hub utilizador
https://github.com/user-attachments/assets/677ec16e-456a-4229-849b-2f0a062bdfdf
###  Calibração 
https://github.com/user-attachments/assets/a69d4c37-c23b-4934-8f15-7ea7e88a25d4
###  Exercicio Agachamento real time
https://github.com/user-attachments/assets/42a02141-1ec1-4bf3-98a5-5dfbe93effd3
###  Resultados exercicio
https://github.com/user-attachments/assets/02eea182-ff08-405c-acd8-cbe90ddb5083
