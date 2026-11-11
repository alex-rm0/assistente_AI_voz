# Estabilização da conversa e do email

Data: 2026-11-11. Ronda baseada no teste de voz e workspace fornecido pelo utilizador.
São correções de comportamento existente, não uma nova fase de funcionalidades.

## Correções

- Afirmações de gravação/criação em inglês sem execução real são bloqueadas nos caminhos conversacionais abrangidos. Texto citado e conteúdo do próprio email não são tratados como ações do assistente.
- A correção de uma inicial do destinatário fica no workflow e pede confirmação antes de alterar o nome. O artefacto mantém a identidade.
- Um pedido para guardar o email já recolhido pode gerar o rascunho e escolher automaticamente um nome de ficheiro. O conteúdo guardado é o rascunho canónico, através da ferramenta existente e das permissões da workspace. Um nome explícito válido continua a ser aceite.
- Uma falha de geração ou gravação não produz confirmação de sucesso. O nome do ficheiro é recuperado do resultado real, não inventado pelo modelo.
- Pedir mais um email cria outro artefacto; guardar não fecha automaticamente a tarefa nem agenda uma reunião.
- O cartão recebe o payload que a ponte JavaScript descartava. Mostra os dados recolhidos, o rascunho e o comprovativo de gravação com `textContent`, sem alterar a geometria. Os rótulos acompanham mudanças explícitas de idioma, sem traduzir o documento guardado.
- Convites como «O que é que podemos fazer hoje?» deixam de ser classificados como conhecimento geral. A orientação existente usa apenas tópicos pessoais recuperados; o guard cobre também as variantes de agenda vazia observadas nas novas corridas.
- A hora explícita é preservada: «9h30 da noite» fornece a restrição 21:30 ao composer. Um rascunho com hora ausente ou contraditória não é aceite para exportação.
- A personalidade ligeiramente machine mantém-se na conversa social. Comentários como `System checks:` não podem tornar-se conteúdo do email.

## Validação inicial

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

## Extensão transversal: gravação, cartões e idioma

- A gravação continua a exigir um pedido do utilizador. Só a escolha do nome passa a ser automática: nomes locais `.txt`, derivados do conteúdo/título, com sufixo numérico em caso de colisão. Não se sobrescrevem ficheiros nem se modificam os originais.
- Cobertura por família: email canónico e legado, férias, comparações, decisões, estudo, pesquisa associada a tarefas, escrita/atividade genérica, documentos revistos, resumos de documentos/PDF e resumos de texto. Um email preparado a partir de documento não reutiliza um email antigo.
- Conteúdo preparado é exportado sem nova geração e sem copiar a conversa de preparação. Se apenas existe conhecimento parcial da tarefa, o ficheiro identifica-o como resumo atual, incluindo lacunas; não se faz passar por um itinerário final.
- A consulta do nome lê o resultado real da gravação. A referência semântica de exportação pertence ao WorkSession e sobrevive a interrupções e mudanças de idioma, sem retomar nem alterar outra tarefa por engano.
- Os cartões mostram conteúdo e referências reais de gravação. A mudança explícita de idioma atualiza os cartões abertos, incluindo os de tarefas em pausa; o conteúdo original dos documentos e os dados introduzidos pelo utilizador não são traduzidos automaticamente.
- Rótulos documentais, resultados fixos de ficheiros e diálogos de capacidade/consentimento respeitam o idioma configurado. Não se alteraram limites, policy, resize ou geometria.
- A verificação cruzada também encontrou um destino explícito que não era reconhecido fora do fallback Estocolmo. A extração de frases como «férias em Coimbra» passou a ser genérica e conservadora, com negativos para meses, família e destinos recusados.
- Testes e provas de Qt desta extensão usam armazenamento temporário. Nenhuma credencial, configuração local ou ficheiro pessoal entra nos commits. Não houve chamadas Anthropic nem push.

### Validação da extensão

- Suite completa final, isolada: **2151 passed, 1 xfailed**, em 97,33 s. Uma corrida anterior terminou sem sumário e não foi usada como aprovação; a repetição isolada completou-se. O xfail conhecido sobre critérios de comparação permanece fora deste âmbito.
- `tests/test_artifact_export.py`: **58 passed**; frontend: **284 passed**. `node --check` e `compileall` passaram.
- Qt real, QWebEngine/QWebChannel e Ollama local: **33 turnos e sete exportações**, sem fornecer nomes de saída. Foram guardados email, plano de férias, comparação, documento revisto, resumo de PDF, email preparado a partir do PDF e outro plano com um email interrompido.
- A validação verificou os ficheiros existentes, o nome devolvido após cada gravação, o novo email de PDF contra o conteúdo canónico correspondente, os originais intactos, o destinatário do email interrompido por preencher e Coimbra presente no plano certo. Nenhum destes testes usa capturas do ecrã ou exige fala do utilizador.
- A confirmação de minimização ao atingir o limite de quatro tarefas foi respeitada, não contornada. A política e os limites continuam iguais.
- Bateria final dos cenários 1–3 e 11–14: passou, código de saída 0. Os 51 ficheiros de `data/` foram novamente repostos com hashes SHA256 idênticos aos do backup.
- Continua a haver limitações da extração por regras e do modelo local. Isto não implementa geração automática de itinerários completos, novos formatos de saída ou gravação sem pedido explícito.
