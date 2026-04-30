# Fitness Tracking App

*Aplicação para monitorização de exercícios em tempo real*  
Trabalho de Final de Curso - Engenharia Informática

---

## Autores

| Número | Nome |
|--------|------|
| 45155 | Rodrigo Correia |
| 51598 | David Delgado |

**Orientador:** [Nome do orientador, se quiseres adicionar]

---

## Links Úteis

- **Repositório principal** (este)
- **Drive de recursos** -> [Microsoft SharePoint](https://iselpt-my.sharepoint.com/:f:/g/personal/a51598_alunos_isel_pt/IgDmtY9-GcMRT6fJk1uhffsZAcqzwMO2dog03hfgFe_jKHI?e=OK5wBT)
- **Figma (UI/UX)** -> *a definir*
- **Documentação técnica** -> `docs/` (a criar)

---

## Estado do Projeto



---

## Tecnologias Utilizadas

| Camada | Tecnologia | Versão |
|--------|------------|--------|
| Frontend (App) | Kotlin + Android | mín. API 24 |
| Backend / Motor | Python → Kotlin (em migração) | - |
| Modelos ML | Ollama (local) | em avaliação |
| Base de Dados | *a definir* | - |

---

## Estrutura do Projeto (Atual)



---

## Modelo de Dados (Proposto)

*Em definição durante a semana de 26 de Abril - 2 de Maio*

```mermaid
erDiagram
    UTILIZADOR ||--o{ SESSAO : realiza
    SESSAO ||--o{ EXERCICIO : contem
    EXERCICIO ||--o{ SERIE : possui
    
    UTILIZADOR {
        string id PK
        string nome
        string email
        int idade
        float peso
    }
    
    SESSAO {
        string id PK
        datetime data_inicio
        datetime data_fim
        int duracao_segundos
    }
    
    EXERCICIO {
        string id PK
        string nome
        string tipo
        string grupo_muscular
    }
