# Echo Roadmap

Data: 2026-07-22  
Base: checkpoint técnico do commit `3f6cca7`

Este roadmap organiza as próximas fases por capacidade cognitiva, não por acumulação de funcionalidades.

## Fase 0 — Checkpoint E Baseline

Objetivo: congelar o estado actual e criar uma base técnica confiável.

Entregáveis:

- `docs/ECHO_CHECKPOINT.md`;
- `docs/ECHO_ARCHITECTURE.md`;
- `docs/ECHO_FEATURE_MATRIX.md`;
- `docs/ECHO_BASELINE.md`;
- `docs/ECHO_ROADMAP.md`;
- baseline limpa de tests/evals.

Dependências:

- ambiente Python funcional;
- Ollama local;
- working tree limpa.

Critérios de conclusão:

- `pytest` passa;
- evals fixed+generated passam;
- metadata com `git_dirty=false`;
- modelo real registado correctamente.

Riscos:

- regressão futura da `.venv` se voltar a depender de Python da Microsoft Store;
- modelo configurado diferente do modelo real se forem usados overrides sem registo;
- dados antigos contaminados.

Não fazer nesta fase:

- Anthropic;
- Ruflo runtime;
- pesquisa web nova;
- voz nova;
- novos workspaces.

## Fase 1 — Comparação De Modelos

Objetivo: saber objectivamente que modelo/provider melhora o Echo.

Entregável:

- provider abstraction usada também pelo runtime;
- baseline comparável entre modelos locais;
- preparação para providers externos sem chamadas pagas por omissão.

Dependências:

- corrigir `settings["ollama"]["model"]`;
- estabilizar evals;
- telemetria limpa.

Critérios de conclusão:

- é possível correr o mesmo conjunto de evals com dois modelos;
- relatório compara latência, pass rate, falhas e qualidade;
- runtime continua funcional com Ollama.

Riscos:

- introduzir custos externos sem controlo;
- misturar provider abstraction com routing.

Não fazer:

- não substituir `AssistantEngine`;
- não introduzir Ruflo;
- não adicionar swarms.

## Fase 2 — Limpeza E Relevância Da Memória

Objetivo: garantir que a memória ajuda o Echo a conhecer o Alexandre sem contaminar respostas.

Entregável:

- auditoria/migração de mojibake;
- política de curadoria;
- separação clara entre factos, hipóteses, logs e respostas antigas;
- deduplicação entre memories, structured facts, timeline, session summaries e Personal Model.

Dependências:

- baseline limpa;
- script de auditoria read-only;
- plano de backup.

Critérios de conclusão:

- memórias antigas técnicas deixam de entrar em prompts normais;
- factos importantes têm source/confidence/status;
- Response Composer nunca copia memória literal;
- Personal Model contém conhecimento útil e curado.

Riscos:

- apagar conhecimento útil;
- transformar hipótese em facto;
- perder continuidade.

Não fazer:

- não migrar tudo automaticamente sem relatório;
- não usar memórias como texto final.

## Fase 3 — Pesquisa Real Com Grounding

Objetivo: ligar uma ferramenta real de pesquisa à arquitectura existente sem falsas promessas.

Entregável:

- ferramenta `web_search` ou equivalente;
- resultados com título, snippet, URL/fonte;
- grounding obrigatório;
- UI research workspace alimentado por resultados reais;
- evals de pesquisa com fontes.

Dependências:

- política de fontes;
- limites de rede;
- segurança de URLs;
- tratamento de ausência de ferramenta.

Critérios de conclusão:

- “pesquisa sobre X” executa ferramenta real;
- se a ferramenta falha, o Echo admite claramente;
- cartões mostram dados grounded;
- o modelo nunca inventa fontes.

Riscos:

- resultados inventados;
- scraping instável;
- dependência externa sem fallback.

Não fazer:

- não gerar cartões falsos;
- não resumir sem fontes;
- não abrir browser sem confirmação quando for ação externa.

## Fase 4 — Primeiro Workflow Diferenciador Completo

Workflow recomendado: “Retomar automaticamente um projecto exactamente do ponto em que ficou”.

Fluxo esperado:

```text
observar sessão
→ guardar decisões, ficheiros e próximo passo
→ fechar
→ retomar mais tarde
→ apresentar contexto relevante
→ permitir abrir projecto e continuar
```

Objetivo: transformar contexto, memória, sessão, tarefas e desktop actions numa experiência única.

Entregável:

- Session Manager curado;
- Context Observer consistente;
- Project Memory inicial;
- pergunta “onde ficámos?” com resposta natural;
- sugestão de próximo passo;
- opção de abrir projecto conhecido com confirmação.

Dependências:

- memória limpa;
- desktop actions seguras;
- Context Observer com summaries úteis.

Critérios de conclusão:

- o Echo identifica o último projecto activo;
- descreve o último trabalho sem logs técnicos;
- sugere próximo passo grounded;
- abre ambiente de trabalho com confirmação;
- evals cobrem continuidade.

Riscos:

- confundir actividade recente com actual;
- inventar próximo passo;
- repetir resumos técnicos.

Não fazer:

- não criar dashboards;
- não abrir apps sem confirmação;
- não usar dados antigos como factos actuais.

## Fase 5 — Workspaces Adaptativos Adicionais

Objetivo: evoluir Echo OS sem transformar a UI em dashboard.

Entregáveis possíveis:

- workspace de projecto/código;
- workspace de documentos;
- workspace de estudo;
- workspace de planeamento.

Dependências:

- UIEventAdapter estável;
- first workflow completo;
- eventos semânticos bem definidos.

Critérios de conclusão:

- cada workspace aparece só quando ajuda;
- UI regressa ao estado natural;
- conversa não é o centro da experiência.

Riscos:

- excesso visual;
- cartões sem grounding;
- duplicar UI clássica.

Não fazer:

- não implementar todos os workspaces de uma vez;
- não mostrar debug em modo normal.

## Fase 6 — Voz

Objetivo: tornar a voz parte natural da presença, não apenas transcrição.

Entregáveis:

- STT robusto;
- estado de voz fiável;
- eventual TTS local/desligável;
- conversa de voz curta e natural;
- wake word só quando houver política de privacidade clara.

Dependências:

- audio device estável;
- Whisper/ffmpeg/microfone validados;
- UI preparada para estados de voz.

Critérios de conclusão:

- clicar, falar, transcrever com boa qualidade;
- erros claros;
- sem envio automático;
- privacidade explícita.

Riscos:

- ruído;
- transcrição errada;
- interrupções constantes.

Não fazer:

- não activar monitorização de voz contínua por omissão;
- não usar ElevenLabs sem opção local/desligável;
- não implementar wake word antes da política.

## Fase 7 — Experiências Externas Com Ruflo

Objetivo: avaliar Ruflo como ferramenta de desenvolvimento, não como runtime.

Entregáveis:

- experiência isolada para gerar candidatos de evals;
- revisão de cobertura;
- análise de falhas;
- relatório custo/benefício.

Dependências:

- licença auditada;
- isolamento fora do runtime;
- sem dependência obrigatória Node.

Critérios de conclusão:

- prova de utilidade concreta;
- zero impacto no runtime;
- outputs revistos por humano.

Riscos:

- complexidade;
- duplicação do Echo;
- confundir ferramenta de desenvolvimento com arquitectura do produto.

Não fazer:

- não colocar Ruflo entre UI e `AssistantEngine`;
- não introduzir swarms no fluxo normal;
- não substituir memória/router/tools.

## Regra Principal

Sempre que surgir uma nova ideia:

> Esta ideia melhora uma capacidade cognitiva existente ou cria apenas uma funcionalidade isolada?

Se for apenas funcionalidade isolada, deve ser adiada.
