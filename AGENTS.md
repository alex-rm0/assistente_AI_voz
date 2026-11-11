# Regras do projeto Echo (AssistenteIA)

CLAUDE.md e AGENTS.md têm o mesmo conteúdo: altera sempre os dois.

## Modo estabilização
- Sem funcionalidades novas. Menor diff possível.
- Diagnóstico primeiro: confirma a causa antes de mudar código.
- Teste que falha primeiro, depois a correção.
- Se a correção passar de 3 ficheiros, pára e mostra o plano antes.
- Um commit por correção. Sem push.
- conversation.py não cresce: código novo vai para um módulo próprio sempre
  que possível (ex.: assistant/artifact_export.py). Cada correção deve manter
  ou reduzir o tamanho do ficheiro; se tiver de crescer, o relatório explica
  porquê.

## Política de idioma
- `response_language` é fixo: só muda com um comando explícito
  ("responde em português" / "responde em inglês"). Com en-GB, toda a
  resposta visível sai em inglês, mesmo que o utilizador fale português.

## Validação
- pytest completo: `.venv\Scripts\python.exe -m pytest -q`.
- `scripts/battery_driver.py` para os cenários 1-3 e 11-14 de
  docs/STABILIZATION_BATTERY.md. Escreve em `data/`: faz backup antes e
  repõe no fim.
- Nunca corras benchmarks de GPU (`scripts/benchmark_stt.py`) em paralelo
  com outro agente: os tempos e a memória da GPU ficam contaminados.

## Trabalho paralelo
- O `stab/paralelo` foi integrado no main (merge `7741fc9`). Nenhuma
  worktree é dona de ficheiros: todos se editam no main.
- Um agente de cada vez no mesmo working tree. Trabalho em paralelo exige
  worktree própria.
- Antes de usar a GPU (benchmarks, STT em GPU), confirma com `nvidia-smi`
  que nenhum outro agente a está a usar.

## Referências
- Antes de propor trabalho novo, ler docs/ECHO_NORTH_STAR.md.
- KNOWN_ISSUES.md: problemas conhecidos e o seu estado.
- docs/STABILIZATION_BATTERY.md: bateria de estabilização.
