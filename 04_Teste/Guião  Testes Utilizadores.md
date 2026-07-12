# GUIÃO DE TESTES COM UTILIZADORES — FITNESS TRACKER
**Objetivo:** Validar a autonomia do utilizador no fluxo inicial (Login/Navegação) e a eficácia do algoritmo de deteção por ângulos (BlazePose) sob restrições de ambiente.

---

## FICHA DE METADADOS (A preencher pelo Observador)
* **Nome utilizador Avaliado:** ______
* **Data:** ___/___/2026
* **Dispositivo:** [ ] Próprio  [ ] Observador
* **Distância da Câmara aprox:** ______ metros
* **Iluminação:** [ ] Boa/Direta  [ ] Fraca/Penumbra  [ ] Contra-luz (Janela atrás)
* **Outros dados (ex: Espaço amplo, roupa escura):** ________________________

---

## 0. FLUXO INICIAL E AUTONOMIA (Sem Intervenção)
*Nota para o Observador: Entregar o dispositivo na página inicial e não dar pistas.*

### Instruções Diretas ao Utilizador:
1. "Por favor, tente entrar na aplicação (fazer login ou criar conta se necessário)."
2. "Explore o menu e tente iniciar uma sessão de treino autónoma com os exercícios disponíveis."

### Registo de Ocorrências (Interface e Navegação):
* **Autenticação (Login/Registo):** [ ] Fluido  [ ] Hesitou  [ ] Bloqueou/Errou
* **Início do Exercício:** [ ] Encontrou o botão sozinho  [ ] Demorou a perceber  [ ] Precisou de ajuda
* **Erros de UX Notados (Ex: botões confusos, falta de feedback visual):**
  ______________________________________________________________________

---

## 1. PROTOCOLO DE OBSERVAÇÃO ATIVA (Deteção e Ângulos)
*Garantir enquadramento de corpo inteiro antes de iniciar a parte física.*

### Instruções Diretas ao Utilizador:
1. "Posicione-se em frente à câmara até que a aplicação o identifique."
2. "Realize uma série de 5 Agachamentos (Squats)."
3. "Realize uma série de 5 Afundos (Lunges), alternando as pernas."
4. "Realize uma série de 5 Flexões (Push-ups)."

### Registo de Ocorrências (Métricas Técnicas):
* **Deteção Inicial:** [ ] Imediata  [ ] Demorada  [ ] Falhou (Exigiu reposicionamento)
* **Fidelidade da Contagem:**
  * **Squats:** [ ] Correta  [ ] Falhou contagem  [ ] Falsos positivos (Fantasmas)
  * **Lunges:** [ ] Correta  [ ] Falhou contagem  [ ] Falsos positivos (Fantasmas)
  * **Push-ups:** [ ] Correta  [ ] Falhou contagem  [ ] Falsos positivos (Fantasmas)
* **Oclusão / Perda de Pontos:** Houve rutura de tracking em algum movimento? Onde?
  ______________________________________________________________________
* **Fatores Críticos Notados:** [ ] Distância insuficiente  [ ] Resolução  [ ] Cor da roupa/Contraste

---

## 2. ENTREVISTA QUALITATIVA PÓS-TESTE
*Formular de modo neutro. Registar respostas curtas.*

1. **Como correu o processo inicial de entrar na app e encontrar o treino? Sentiu alguma confusão?**
   
   R: __________________________________________________________________

3. **Como descreve a experiência de deteção do corpo pela câmara?**
   
   R: __________________________________________________________________

5. **Sentiu necessidade de alterar o ritmo ou a amplitude do movimento para a app contar?**
   
   R: __________________________________________________________________

7. **Encontrou dificuldades no posicionamento inicial do dispositivo ou com a luz ambiente?**
   
   R: __________________________________________________________________

---

## 3. QUESTIONÁRIO DE USABILIDADE SIMPLIFICADO (SUS)
*Solicitar ao utilizador que classifique de 1 (Discordo Totalmente) a 5 (Concordo Totalmente).*

Forms (Avaliação SuS) -> https://docs.google.com/forms/d/e/1FAIpQLSeMdEUKaM967SH-1HRSSQsBR04-Mcq67C3s2UG5aFf0qPgrmg/viewform?usp=sharing&ouid=111181290584913837555


Perguntas a adicionar ao SUS

- 'Acho que precisaria da ajuda de outra pessoa para posicionar a câmara corretamente.'
- 'Achei que o conjunto de exercícios estava bem estruturado, com intervalos de descanso adequados.'
- 'Achei que houve inconsistência na contagem das minhas repetições.'
- 'Achei que a pontuação dos meus afundos (lunges) e agachamentos (squats) foi fiel ao meu esforço real.'
