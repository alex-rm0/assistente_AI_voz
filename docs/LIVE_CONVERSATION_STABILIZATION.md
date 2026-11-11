# Estabilização da conversa e do email

Data: 2026-11-11. Ronda baseada no teste de voz e workspace fornecido pelo utilizador.
São correções de comportamento existente, não uma nova fase de funcionalidades.

## Correções

- Afirmações de gravação/criação em inglês sem execução real são bloqueadas nos caminhos conversacionais abrangidos. Texto citado e conteúdo do próprio email não são tratados como ações do assistente.
- A correção de uma inicial do destinatário fica no workflow e pede confirmação antes de alterar o nome. O artefacto mantém a identidade.
- Um pedido para guardar o email já recolhido pode gerar o rascunho e pedir o nome do ficheiro. O conteúdo guardado é o rascunho canónico, através da ferramenta existente e das permissões da workspace.
- Uma falha de geração ou gravação não produz confirmação de sucesso. O nome do ficheiro é recuperado do resultado real, não inventado pelo modelo.
- Pedir mais um email cria outro artefacto; guardar não fecha automaticamente a tarefa nem agenda uma reunião.
- O cartão recebe o payload que a ponte JavaScript descartava. Mostra os dados recolhidos, o rascunho e o comprovativo de gravação com `textContent`, sem alterar a geometria. Os rótulos acompanham mudanças explícitas de idioma, sem traduzir o documento guardado.
- Convites como «O que é que podemos fazer hoje?» deixam de ser classificados como conhecimento geral. A orientação existente usa apenas tópicos pessoais recuperados; o guard cobre também as variantes de agenda vazia observadas nas novas corridas.
- A hora explícita é preservada: «9h30 da noite» fornece a restrição 21:30 ao composer. Um rascunho com hora ausente ou contraditória não é aceite para exportação.
- A personalidade ligeiramente machine mantém-se na conversa social. Comentários como `System checks:` não podem tornar-se conteúdo do email.

## Validação

- Suite completa: **2086 passed, 1 xfailed**, em 95,69 s. O xfail conhecido sobre critérios de comparação fora da lista fixa permanece fora desta ronda.
- Frontend: **282 passed** em `tests/test_web_ui_prototype.py`.
- `node --check` nos dois ficheiros JavaScript alterados e `compileall` passaram.
- Qt real, QWebEngine e QWebChannel, com Ollama local: 12 turnos escritos reproduziram a correção do nome, geração, gravação, consulta do nome, explicação da tarefa ainda aberta e criação de outro email. O ficheiro temporário correspondeu ao rascunho canónico; os passos determinísticos não chamaram o LLM. A geração do email usou uma chamada.
- Bateria prescrita pelo AGENTS.md: cenários 1–3 e 11–14 passaram. Nesta corrida, UI pronta aos 1130 ms, saudação em 105 ms e fecho completo em 38 ms, durante o prewarm do TTS. Estes são tempos do driver offscreen, não um benchmark geral de arranque.
- Os 51 ficheiros de `data/` foram copiados antes da bateria, repostos depois e verificados por SHA256. Os testes de exportação usaram uma workspace temporária.
- Sem chamadas Anthropic, sem alteração de `config/settings.json`, ficheiros pessoais da workspace ou providers. Sem push.

## Limites e pendências

- Não houve nova gravação pelo microfone nem validação humana da fala nesta ronda. A validação Qt foi escrita e instrumentada, sem captura do ecrã.
- A latência inicial da voz continua em aberto: no log fornecido, o primeiro turno teve 4756 ms de carregamento STT, 1346 ms de transcrição, 4711 ms no engine e 919 ms até ao primeiro áudio TTS; o total reportado foi 12777 ms. As transcrições seguintes ficaram muito mais rápidas. Não se introduziu outro prewarm nesta correção.
- Os guards são determinísticos e limitados aos padrões cobertos. Não constituem uma garantia universal contra todas as paráfrases ou sugestões desadequadas do modelo.
- O híbrido `PendingWritingTask`/`ActiveTaskContext` e a qualidade geral do modelo continuam a ser dívida existente; não foram refatorados.
