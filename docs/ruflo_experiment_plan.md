# Plano de experiência: Ruflo como ferramenta de geração/revisão de testes

Este documento propõe uma experiência futura, isolada, para avaliar se o Ruflo tem
utilidade como ferramenta de desenvolvimento — especificamente para gerar e rever
candidatos de casos de avaliação (`evals/cases/candidates/`). Não descreve nem autoriza
qualquer integração no runtime do Echo.

## Estado atual

Esta tarefa (estabilizar → medir → separar o modelo) foi concluída sem tocar no Ruflo.
A infraestrutura de evals (`evals/`) já existe, é independente da UI, e já produz
relatórios estruturados. O Ruflo não participou em nada disto e não deve participar
enquanto este plano não for explicitamente aprovado e executado numa branch própria.

## Âmbito da experiência

```text
branch: experiment/ruflo-evals
objetivo: usar o Ruflo apenas para gerar e rever candidatos de teste em evals/cases/candidates/
runtime do Echo: intocado (assistant/, ui/, app.py não mudam)
dados pessoais: proibidos (o Ruflo nunca vê data/, nem os ficheiros reais do utilizador)
alterações de produção: proibidas (nenhum commit desta branch é elegível para merge
  direto em main sem revisão humana explícita)
```

## Pré-requisitos

1. A infraestrutura de evals (`evals/`) tem de estar estável e a correr sem exceções —
   já verificado nesta tarefa (`python -m evals.run_evals`, 33/33 casos, 0 exceções).
2. A branch `experiment/ruflo-evals` só pode ser criada a partir de um estado em que
   `python -m pytest` não tem regressões novas face ao `main`.
3. O Ruflo corre isolado do ambiente de dados reais — só vê o código-fonte de
   `assistant/` (para entender a arquitetura) e `evals/` (para gerar casos), nunca
   `data/`, `config/settings.json` com segredos, ou qualquer histórico de conversa real.
4. Existe um humano disponível para rever cada candidato gerado antes de qualquer
   candidato ser promovido de `evals/cases/candidates/` para `evals/cases/fixed/`.

## Tarefa inicial (não destrutiva)

A primeira e única tarefa a dar ao Ruflo nesta experiência:

```text
Analisa a infraestrutura de evals do Echo.
Identifica lacunas de cobertura.
Gera apenas candidatos de teste.
Não alteres código de produção.
Não uses dados pessoais.
Produz um relatório.
```

Critério de conclusão desta tarefa inicial: um relatório de texto + ficheiros JSON em
`evals/cases/candidates/` (nunca em `fixed/` ou `generated/`), sem qualquer alteração
a ficheiros fora de `evals/cases/candidates/` e `docs/`.

## Riscos

- **Alucinação de comportamento esperado**: o Ruflo pode gerar candidatos com
  `expected` incorreto (assumir um comportamento que o Echo não tem). Mitigação: todo
  o output do Ruflo entra em `candidates/`, nunca em `fixed/`, e carrega
  `review_status: "unreviewed"` — nada corre automaticamente na suite principal.
- **Exposição de dados**: se o Ruflo tiver acesso de leitura amplo ao repositório,
  pode encontrar segredos em `config/settings.json` ou dados reais em `data/`.
  Mitigação: a branch de experiência corre com esses caminhos excluídos do contexto
  dado ao Ruflo (via `.gitignore`-style exclusão explícita na configuração da
  experiência, não apenas confiança implícita).
- **Scope creep**: a tentação de deixar o Ruflo "só ajustar uma coisinha" no código de
  produção. Mitigação: a tarefa inicial proíbe-o explicitamente, e qualquer PR desta
  branch para `main` exige revisão humana linha a linha antes do merge.
- **Custo/tempo desproporcional**: gerar candidatos de baixa qualidade que exigem mais
  tempo de revisão humana do que os candidatos template-based já existentes
  (`evals/generate_cases.py`). Mitigação: o critério de abandono abaixo cobre isto.

## Critérios de sucesso

1. O Ruflo produz pelo menos alguns candidatos genuinamente novos (não duplicados dos
   já existentes em `fixed/`/`generated/`) que, após revisão humana, revelam uma lacuna
   real de cobertura (uma categoria, frase ou cenário que os 33 casos fixos e as 8
   variações geradas não cobrem).
2. Nenhuma alteração fora de `evals/cases/candidates/` (e opcionalmente `docs/`) é
   produzida sem aprovação humana explícita.
3. O tempo de revisão humana por candidato é razoável (ordem de minutos, não de
   dezenas de minutos por caso) — caso contrário a ferramenta não compensa o esforço.

## Critérios de abandono

1. O Ruflo tenta ou sugere alterar código de produção (`assistant/`, `ui/`, `app.py`,
   `config/`) apesar da instrução explícita em contrário.
2. O Ruflo acede ou tenta aceder a `data/` ou a qualquer dado pessoal real.
3. A maioria dos candidatos gerados não sobrevive à revisão humana (assertions
   incorretas, cenários irrealistas, duplicação do que já existe).
4. O esforço de revisão supera claramente o valor dos candidatos obtidos, comparado
   com continuar a expandir `evals/generate_cases.py` manualmente.

Se qualquer critério de abandono se verificar, a branch `experiment/ruflo-evals` é
descartada (não faz merge) e este documento é atualizado com o que se aprendeu.

## Fora de âmbito desta experiência

- Migrar memória para o Ruflo.
- Substituir o router (fast_router / executive_function / intent_engine).
- Criar agentes ou swarms.
- Qualquer alteração ao comportamento de produção do Echo.

Estas exclusões repetem deliberadamente a instrução original da Parte 4 — não são
apenas um resumo, são a condição para esta experiência ser aprovada.
