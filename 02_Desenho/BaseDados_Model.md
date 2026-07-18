# Modelo e Estrutura da Base de Dados (Cloud Firestore NoSQL)

Este documento resume a organização dos dados no Cloud Firestore para a aplicação Fitness Tracker, suportando a sincronização em tempo real e a estratégia de agregação na escrita (*Write-Time Aggregation*).

---

##  Estrutura e Hierarquia da Base de Dados

O Cloud Firestore utiliza uma estrutura NoSQL hierárquica baseada em Coleções (pastas) e Documentos (ficheiros JSON). As coleções `workouts` (histórico de treinos) e `custom_plans` (planos criados pelo utilizador) são subcoleções aninhadas sob o documento do respetivo utilizador. Isto isola os dados de cada atleta por razões de desempenho e privacidade.

```mermaid
graph TD
    usersCol[(Coleção: users)] --> userDoc{Documento: users/userId}
    userDoc --> userFields["Campos de Perfil (Agregados Lifetime/Semana):
                             - name: String
                             - numericId: String
                             - level: Int
                             - xpPoints: Int
                             - totalKcal: Double
                             - overallCadenceStability: Double
                             - weeklyKcal: Double
                             - weeklyCadenceStability: Double
                             - weeklyWorkouts: Int
                             - lastWeeklyReset: Timestamp"]
    
    userDoc --> workoutsCol[(Sub-Coleção: workouts)]
    workoutsCol --> workoutDoc{Documento: workouts/workoutId}
    workoutDoc --> workoutFields["Detalhes do Treino:
                                   - workoutName: String
                                   - date: Timestamp
                                   - durationSeconds: Int
                                   - caloriesBurned: Double
                                   - totalReps: Int
                                   - averageFormScore: Int
                                   - weightKg: Double
                                   - volume: Double
                                   - cadenceScore: Double"]
    
    userDoc --> plansCol[(Sub-Coleção: custom_plans)]
    plansCol --> planDoc{Documento: custom_plans/planId}
    planDoc --> planFields["Configuração do Plano:
                                 - planName: String
                                 - createdAt: Timestamp
                                 - stepsJson: String (Array JSON de passos)"]
    
    friendRequestsCol[(Coleção: friend_requests)] --> requestDoc{Documento: friend_requests/requestId}
    requestDoc --> requestFields["Pedido de Amizade:
                                   - senderUid: String
                                   - receiverUid: String
                                   - status: String (PENDING/ACCEPTED)
                                   - timestamp: Timestamp"]
```

---

##  Estratégia de Agregação na Escrita (Write-Time Aggregation)

Para manter a renderização de **Tabelas de Classificação (Leaderboards)** e dos **Gráficos de Progresso** rápidos ($O(1)$) e de baixo tráfego de rede, a aplicação atualiza atomicamente os agregados e o XP no documento do utilizador no exato momento em que um treino é submetido.

```mermaid
sequenceDiagram
    participant App as Aplicação Android (Kotlin)
    participant DB as Cloud Firestore (NoSQL)
    App->>DB: Iniciar Transação Atómica
    App->>DB: Escrever registo detalhado de treino em /workouts/
    App->>DB: Ler documento de perfil /users/{uid}/
    App->>DB: Validar data de Reset Semanal (último reset vs agora)
    alt Se for semana diferente
        App->>App: Resetar acumuladores semanais (weeklyKcal, weeklyWorkouts, weeklyCadence)
    end
    App->>App: Calcular novos acumuladores Lifetime e Semanais
    App->>DB: Atualizar campos agregados e XP no documento do utilizador
    DB-->>App: Transação Concluída com Sucesso
    App->>App: Atualizar Compose UI (Dashboard & Ladders) instantaneamente
```

---

##  Instruções para Limpeza Manual de Dados (Ambiente de Testes)

Durante os ensaios de usabilidade com os utilizadores, para limpar dados antigos mantendo apenas os treinos efetuados no dia de hoje:
1. Aceda à consola do **Firebase Console** e clique em **Firestore Database**.
2. Abra a coleção `users` e procure o ID do seu utilizador.
3. Entre na subcoleção **`workouts`**.
4. Apague os documentos cujo campo `date` seja anterior ao dia de hoje.
