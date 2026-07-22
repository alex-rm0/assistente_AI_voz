# Echo evals

Infraestrutura independente de avaliação do motor do Echo (`assistant.conversation.AssistantEngine`).
Não está acoplada à UI — chama `engine.respond()` diretamente e lê a telemetria estruturada de
`engine.get_last_turn_telemetry()`, nunca faz parsing do `[TURN TRACE]` impresso no terminal.

## Estrutura

```
evals/
├── cases/
│   ├── fixed/       casos aprovados, corridos por omissão (inclui real_conversation/)
│   ├── generated/   variações geradas por template (secção 2.8) — só corre com --include-generated
│   └── candidates/  extraídos de logs reais (secção 2.9) — nunca corridos automaticamente
├── results/
│   ├── latest/                       cópia da execução mais recente (report.json/.csv/.md + metadata.json)
│   ├── runs/YYYY-MM-DD/<run_name>/   uma pasta por execução (ver convenção de nomes abaixo)
│   ├── comparisons/<provider>__<model>/<comparison_id>/  agregados de --repeat (secção 1.3 / Parte 7)
│   ├── baselines/<run_name>/         execuções marcadas com --mark-baseline (nunca podadas)
│   ├── legacy/                       relatórios do formato antigo (achatado), preservados sem alterações
│   └── index.md                      uma linha por execução, mais recente primeiro
├── schemas.py             EvalCase, TurnExpectation, TurnResult, TurnEvaluation, CaseEvaluation, ...
├── harness.py             isolamento (ECHO_ENV=test) + construção do motor por caso
├── assertions.py          todas as verificações determinísticas (secção 2.4)
├── failure_classifier.py  classificação preliminar de falhas (secção 2.6)
├── human_review.py        heurísticas de sinalização para revisão humana (Parte 6) — nunca bloqueiam
├── comparisons.py         PASS_STABLE / FAIL_STABLE / FLAKY / NOT_RUN entre repetições (Parte 7)
├── results_store.py       layout de diretórios, metadata.json, retenção, baseline, index.md (Parte 1)
├── report.py              construção do CONTEÚDO dos relatórios JSON/CSV/Markdown (secção 2.5)
├── generate_cases.py      gerador por template + extração de candidatos a partir de logs
└── run_evals.py           CLI principal
```

### Convenção de nomes de uma execução

```
<timestamp>__<suite>__<provider>__<model>__r<repeat>
2026-07-19_19-00-01__fixed-generated__ollama__llama3.1-8b__r1
```

`suite` é `fixed` por omissão, `category-<nome>` quando `--category` é usado, ou
`case-<id>` quando `--case` é usado; `-generated` é acrescentado quando `--include-generated`
está ativo. `metadata.json` nessa pasta guarda `run_id`, `timestamp`, `provider`, `model`,
`categories`, `included_generated`, `repeat`, `git_commit`, `git_dirty`, `total_cases`,
`passed`, `failed`, `exceptions`, `average_latency_ms`, `command_used` e `baseline`.

## Isolamento (nunca toca nos dados reais)

Cada caso recebe um motor `AssistantEngine` totalmente novo, com o seu próprio diretório
temporário (`ConversationMemory` + `LongTermMemory` isolados). O runner define
`ECHO_ENV=test` e `ECHO_TEST_DATA_DIR=<diretório temporário>` no arranque e apaga esse
diretório no fim (a menos que `--keep-data` seja passado). Os dados pessoais reais em
`data/` nunca são lidos nem escritos por um eval.

## Formato de um caso

```json
{
  "id": "memory_exam_recall_001",
  "category": "memory",
  "description": "...",
  "setup": [{"fact_type": "academic_event", "fields": {"discipline": "Estratégias Algorítmicas"}}],
  "clear_conversation_before": true,
  "turns": [
    {"user": "Que exame vou ter para a semana?", "expected": {"selected_path": "MEMORY_RECALL", "llm_calls_max": 0, "must_contain": ["Estratégias Algorítmicas"]}}
  ],
  "tags": ["regression", "grounding"]
}
```

`setup` aceita dois tipos de passo:
- `{"say": "mensagem"}` — um turno de conversa real (não avaliado), útil para testar extração passiva.
- `{"fact_type": "...", "fields": {...}}` — escreve um facto estruturado diretamente na
  memória de longo prazo isolada do caso, sem depender do LLM para o extrair.

## Assertions suportadas (`expected`)

`selected_path`, `forbidden_paths`, `llm_calls_min`, `llm_calls_max`, `expected_tools`,
`forbidden_tools`, `no_tools_used` (equivalente a `tools_used == []`), `forbidden_contexts`
(contextos do `ContextManager` que não podem estar ativos, ex. `["TECH_CONTEXT"]`),
`expected_memory_ids`, `memory_write_action` (usar `null` para "não deve escrever"),
`must_contain`, `must_not_contain`, `response_not_empty`, `response_grounded`, `max_latency_ms`,
`expected_exception`, `max_questions`. Por omissão, TODOS os casos já verificam
`forbid_brazilian_portuguese`, `forbid_unsupported_tool_claim` e `forbid_unsupported_memory_claim`
— só é preciso mencioná-los para os desligar.

## Revisão humana (Parte 6)

Cada `TurnEvaluation` tem `human_review_required` / `review_reasons` / `human_scores`.
`evals/human_review.py` sinaliza automaticamente (nunca bloqueia) respostas com:
afirmação de memória ou ferramenta sem grounding, uma frase de "entidade importante" sem
fonte, entusiasmo excessivo, ou uma pergunta que já tinha resposta no histórico da própria
conversa. `human_scores` (naturalness/context_following/pt_pt_quality/usefulness/
unsupported_assumptions) fica `None` até uma pessoa o preencher — nada neste repositório
escreve nesse campo automaticamente.

## Comandos

```powershell
python -m evals.run_evals
python -m evals.run_evals --category memory
python -m evals.run_evals --category real_conversation
python -m evals.run_evals --case memory_exam_recall_001
python -m evals.run_evals --provider ollama --model llama3.1:8b
python -m evals.run_evals --include-generated
python -m evals.run_evals --repeat 3
python -m evals.run_evals --fail-fast --strict
python -m evals.run_evals --keep-runs 20
python -m evals.run_evals --mark-baseline
python -m evals.generate_cases
python -m evals.generate_cases --from-logs caminho/para/log.txt
```

`--repeat N` corre cada caso N vezes seguidas, guarda uma pasta de execução por repetição, e
gera um agregado em `results/comparisons/<provider>__<model>/<comparison_id>/` com o estado
de cada caso: `PASS_STABLE`, `FAIL_STABLE`, `FLAKY` ou `NOT_RUN` (Parte 7). Um caso `FLAKY`
nunca deve ser lido como "passou" só porque uma repetição teve sucesso — o relatório mostra
sempre a taxa (`2/3`), nunca só o resultado da última repetição.

`--keep-runs N` (por omissão 20) mantém as N execuções mais recentes por combinação
provider/modelo/suite; nunca apaga `latest/`, `comparisons/`, nem execuções marcadas com
`--mark-baseline` (essas também ficam copiadas em `results/baselines/`).

## Código de saída

Por omissão, `run_evals.py` só devolve código de saída ≠0 quando existe pelo menos uma
falha classificada como `INFRASTRUCTURE` (uma exceção Python real). Com `--strict`,
qualquer turno falhado faz o processo sair com código 1.

## Classificação de falhas

Cada turno falhado recebe uma classificação preliminar (`ROUTING`, `MEMORY`, `TOOL`,
`MODEL_OUTPUT`, `LANGUAGE`, `PRESENTATION`, `INFRASTRUCTURE` ou `UNKNOWN`), derivada de
QUAL assertion falhou — não do texto da resposta. É deliberadamente só um ponto de partida
para revisão manual, não um veredito final.

## Fornecedor do modelo

Só `--provider ollama` está implementado nesta tarefa (ver Parte 3 do pedido original).
`harness.build_llm()` é o único ponto que sabe construir um cliente por fornecedor — quando
um `AnthropicProvider`/`OpenAIProvider` existir, só esta função precisa de mudar.

## Limitações conhecidas

- `expected_exception` só é realmente exercitável de forma determinística através dos
  testes unitários em `tests/test_error_handling.py` (que forçam a exceção via
  monkeypatch); não há forma prática de garantir uma exceção real do Ollama a partir
  apenas do texto de um caso JSON.
- As assertions não verificam o VALOR de um campo estruturado (ex.: `status=failed`)
  diretamente — só se uma escrita aconteceu (`memory_write_action`) e o texto da resposta
  seguinte. Uma verificação de campo estruturado explícita fica para uma iteração futura.
- A normalização lazy de registos antigos sem colunas `_raw` (ver tarefa de memória
  anterior) continua por implementar; não é exercitada por nenhum caso aqui.
