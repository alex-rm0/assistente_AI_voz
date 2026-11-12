# Echo — North Star

Data: 2026-11-11 (revisto). Este documento é a **constituição arquitetural** do Echo: diz para onde
vamos e que contratos e princípios não se negoceiam. Os outros docs (ROADMAP, ECHO_ROADMAP,
KNOWN_ISSUES, bateria) descrevem passos; este descreve **o destino e as regras do caminho**.
Agentes (Claude Code, Codex): leiam isto antes de propor trabalho novo. Uma proposta que
contradiga este documento tem de ser discutida com o Alexandre antes de ser implementada.

## 1. O destino

O Echo é uma **presença pessoal contínua** do Alexandre:

- **Está sempre disponível**, mesmo sem janela aberta (ver §1.1 sobre o que "24/7" significa hoje).
- **Acompanha o mundo do Alexandre**, não só o PC: emails, calendário, tarefas,
  as páginas web que ele criou, projetos em curso.
- **Fala com ele onde ele estiver**: por voz e workspace no PC; por mensagem no telemóvel.
- **Interrompe pouco e bem**: avisa só o que importa, quando importa; o resto vai para resumos.
- **Conhece-o** (perfil e memória curados) e **nunca inventa** (grounding acima de fluidez).
- **Age com confirmação por omissão**: lê o que lhe for autorizado; ações com efeito externo pedem
  confirmação, exceto categorias reversíveis às quais o Alexandre deu autorização prévia (§8).

Não é: um chatbot, um wrapper de LLM, nem uma framework genérica (OpenJarvis/Ruflo: referências,
não dependências). Também não é um agente autónomo de propósito geral: não define objetivos por si
nem age fora da autoridade dada; a autonomia existe apenas dentro de categorias explicitamente
autorizadas pelo Alexandre (§8).

### 1.1 O que "24/7" significa

**Destino: disponibilidade 24/7 efetiva. Fase inicial: Core persistente enquanto o PC anfitrião
estiver disponível.** Enquanto o Core correr só no PC, a disponibilidade depende de o PC estar
ligado, acordado, com rede e saudável. A arquitetura **não pode assumir** que o Core fica preso a
esse PC Windows: a passagem futura para híbrido (vigias numa máquina pequena ou na cloud, voz e
workspace no PC) tem de continuar possível sem reescrever o Core.

### 1.2 Invariantes de produto

1. **O Echo não compete pela atenção do Alexandre.** Sem alertas duplicados, sem interrupções
   triviais, resumo quando chega, horas de silêncio respeitadas.
2. **O Echo nunca apresenta inferência como facto.** Grounding acima de fluidez; distinção entre
   fonte e dedução; incerteza dita quando existe; nunca inventar continuidade, memória ou estado
   do mundo.

Ambos devem tornar-se critérios de aceitação testáveis (bateria/evals).

## 2. Arquitetura alvo

```
┌──────────────────────── ECHO CORE (um serviço, módulos internos) ───────────────────────┐
│                                                                                         │
│  Agendador → Vigias ──► EchoEvent ──► Event Store ──► Política de atenção ──► Entrega   │
│  (sites, calendário,     (normalizado,                (INTERRUPT / DIGEST /   (aviso,   │
│   email, builds, tasks)   deduplicado)                 RECORD_ONLY / DROP)    resumo)   │
│                                                                                         │
│  Pedido do utilizador ──► Conversa / Tasks ──► Proposta de ação ──► Política de         │
│  (voz, janela, Telegram)                                            permissão/autonomia │
│                                                        Executor ◄────────┘              │
│                                                           └──► Action Ledger            │
│                                                                                         │
│  Transversais: Memória & perfil · Router de modelos (local/cloud, verifica privacy_class)│
│                Segredos · Saúde (healthy / degraded / unavailable)                      │
└───────────────┬────────────────────────────────┬──────────────────────────┬─────────────┘
                │ API do Core                     │ gateway remoto           │ estado
                ▼                                 ▼                          ▼
      Echo OS no PC (janela, voz,          Telegram (bidirecional,     Painel de saúde
      workspace, wake word, TTS)           allowlist)                  ("o Echo está vivo?")
```

### 2.1 Princípios técnicos

1. **Núcleo separado da interface.** Janela, voz e Telegram são clientes do Core; fechar a janela
   não desliga o Echo. Hoje tudo vive no processo Qt — é a mudança estrutural principal.
2. **Monólito modular primeiro.** O Core começa como **um único serviço/processo** com módulos
   internos claros (agendador, vigias, event store, atenção, memória, conversa, ferramentas/ações,
   gateway remoto, saúde). Nada de microserviços antes do tempo: separar processos só quando houver
   um requisito concreto de fiabilidade, deployment ou escala.
3. **Vigias observam, não notificam.** Um vigia lê estado externo e emite `EchoEvent`; nunca fala
   diretamente com o Alexandre. O resto do Core só consome `EchoEvent`, nunca objetos específicos
   de um fornecedor.
4. **Facto → decisão → entrega, em camadas separadas.** O `EchoEvent` descreve o que aconteceu ou
   foi observado; a `AttentionDecision` descreve o que o Echo decidiu fazer com isso; a entrega
   (`OutgoingNotification`) executa essa decisão. Nenhuma camada absorve a seguinte.
5. **Event Store não é memória.** O Event Store regista eventos operacionais e do mundo. Não é
   memória pessoal de longo prazo e não substitui a memória curada, o perfil nem o estado das
   tasks. O histórico de eventos pode alimentar a curadoria da memória mais tarde, mas guardar um
   evento não equivale a criar uma memória.
6. **Vigias baratos e determinísticos.** Verificar um site, ler a agenda ou listar emails novos
   não precisa de LLM. O LLM só entra para classificar importância e escrever resumos.
7. **Um único sítio decide interromper**: a política de atenção, com regras explícitas e testáveis.
8. **Privacidade verificada na fronteira.** O router de modelos verifica
   `provider_allowed(privacy_class)` antes de enviar qualquer dado a um modelo (§6).
9. **Credenciais nunca no código nem no git** (secret_storage); permissões mínimas, só leitura primeiro.
10. **Falhar parcialmente, nunca globalmente.** Um vigia, fornecedor ou canal que falha não derruba o
   Core. Cada componente principal tem estado de saúde `healthy` / `degraded` / `unavailable`
    (ex.: Ollama em baixo, Claude indisponível, sem internet, Telegram em baixo, exceção num vigia,
    calendário indisponível). O painel de saúde mostra-o.
11. **Fiabilidade antes de capacidade**: reinício automático, logs, testes e bateria. Um assistente
    24/7 que crasha às 3h não serve.
12. **Cérebro trocável**: tudo passa pelo ModelProvider; nenhuma lógica depende de um modelo.

## 3. Contratos do Core

Definidos conceptualmente aqui; tipos e formatos concretos ficam para a F0.

### 3.1 EchoEvent

Modelo canónico de tudo o que acontece fora da conversa. Descreve **só o facto observado**.

| Campo | Significado |
|---|---|
| `id`, `type`, `source` | Identidade; tipo (`site.down`, `site.recovered`, `calendar.upcoming`, `email.received`, `build.failed`, `task.overdue`…); vigia de origem |
| `occurred_at`, `observed_at` | Quando aconteceu / quando o Echo viu |
| `subject`, `payload` | Sobre o quê; detalhes |
| `importance`, `confidence` | Quão importante; quão certo |
| `dedupe_key`, `status` | Chave de deduplicação; estado do evento |
| `privacy_class` | Classe dos dados deste evento (§6) |
| `related_task_id`, `related_project_id` | Ligações a tasks/projetos |

O que o Echo decide fazer com o evento não faz parte dele: vive numa `AttentionDecision` separada,
que referencia o evento pelo `id` (§4).

### 3.2 Idempotência dos vigias

**Os vigias observam estado; o Core emite eventos nas transições relevantes.**

| Estado anterior → atual | Resultado |
|---|---|
| `site.up` → `site.down` | evento `site.down` |
| `site.down` → `site.down` | nada de novo |
| `site.down` → `site.up` | evento `site.recovered` |

A mesma condição externa nunca gera notificações repetidas a cada ciclo. Vale para sites, email,
calendário, deploys/builds e tasks.

### 3.3 Contrato de canal

O Core não depende de tipos de mensagem de nenhum canal. Cada canal (Telegram primeiro; mais tarde
email, app móvel, etc.) é um **ChannelAdapter** que traduz de e para dois contratos:

| Contrato | Campos (conceptuais) |
|---|---|
| `IncomingMessage` | `id`, `source_channel`, `sender_id`, `text`, anexos/referência, `received_at`, `privacy_class` |
| `OutgoingNotification` | `id`, `target_channel`, `recipient`, conteúdo/referência, `priority`, `created_at` |

É só uma fronteira, não uma framework de plugins. Campos e tipos exatos ficam para a F0.

### 3.4 Outros contratos (a definir na F0)

`AttentionDecision`, `ActionRequest`, `ActionResult`, `ActionRecord` (§8.2), classes de
privacidade (§6), modelo de permissões/autonomia (§8), fronteira da API do Core e modelo de saúde.

## 4. Política de atenção

Cada evento recebe uma `AttentionDecision` (separada do evento, §2.1 e §3.1):

| Decisão | Significado |
|---|---|
| `INTERRUPT` | Avisar agora, no canal adequado |
| `DIGEST` | Relevante, mas vai para o próximo resumo |
| `RECORD_ONLY` | Guardado e consultável, sem aviso |
| `DROP` | Irrelevante; não aparece |

Um evento relevante que não deve interromper agora (`DIGEST`) é diferente de um evento irrelevante
(`DROP`). **As horas de silêncio mudam o momento da entrega, não a importância**: um `INTERRUPT`
em horas de silêncio passa a entrega diferida, salvo classes de evento com `override_quiet_hours`.
Horas concretas e classes que furam o silêncio: em aberto (F2).

## 5. Conhecimento do Alexandre

- **Perfil-núcleo pequeno, sempre presente**: nome, idioma de resposta, fuso horário, preferências
  de interação e poucos invariantes estáveis.
- **Restante contexto pessoal recuperado por relevância** (compatível com a Presence / personal
  relevance existente). Nunca meter o perfil inteiro em todos os prompts.
- Memória curada: guardar o que é recorrente, útil e autorizado; nunca texto bruto antigo como facto.

## 6. Classes de privacidade

A classe é atribuída **aos dados concretos** (objeto, conteúdo), não ao módulo nem ao tipo de
fonte. A mesma fonte pode produzir classes diferentes: uma conversa sobre uma preferência banal é
`PERSONAL`; uma conversa pessoal sensível é `PRIVATE`; uma conversa com informação de trabalho
restrita é `RESTRICTED`.

| Classe | Exemplos de dados | Modelos permitidos |
|---|---|---|
| `PUBLIC` | conhecimento geral, página web pública | local e cloud |
| `PERSONAL` | preferência comum, tarefa banal, perfil-núcleo | local e cloud (dentro do orçamento) |
| `PRIVATE` | conversa pessoal sensível, documento pessoal do workspace | local; cloud só por regra explícita |
| `RESTRICTED` | corpo e assunto de emails, informação de trabalho restrita | **só local** |

Regras:

- Cada objeto de dados, `EchoEvent` e `IncomingMessage` transporta `privacy_class`.
- `provider_allowed(privacy_class)` é verificado **na fronteira do routing**, não em cada chamador.
- **Conteúdo derivado herda a classe da origem.** Um resumo local de conteúdo `RESTRICTED` continua
  `RESTRICTED`; só uma regra deliberada e explícita o pode reclassificar.

## 7. Canal remoto: Telegram

Escolhido para a F3 porque é bidirecional, tem API simples, faz push, evita construir uma app
móvel e dá uma superfície de conversa clara.

Requisitos de segurança:

- token do bot só em secret_storage;
- `chat_id`/identidade em allowlist; qualquer chat desconhecido é rejeitado;
- todo o pedido remoto preserva `source_channel`;
- os canais não têm necessariamente a mesma confiança: a política de autorização pode olhar para
  `source_channel` (ex.: certas ações só a partir do PC).

O email pode entrar mais tarde como fonte vigiada, canal de entrega de resumos ou canal secundário
de interação; o Telegram continua a ser o canal conversacional principal da F3.

## 8. Ações, autonomia e registo

### 8.1 Autonomia configurável

**Ações com efeito externo pedem confirmação por omissão.** O Alexandre pode dar **autorização
prévia** a categorias específicas de ações reversíveis. O nível 3 não é um bypass da confirmação: é
uma autorização dada antes.

Cada tipo de ação tem: `autonomy_level`, `max_autonomy_level`, `reversible`, `external_effect`,
`risk_class`.

| Nível | O Echo… | Exemplo |
|---|---|---|
| 0 — Desligado | não faz nem sugere | — |
| 1 — Avisa | só informa | "O site X está em baixo." |
| 2 — Sugere | prepara a ação e pede sim/não | rascunho de resposta a um email |
| 3 — Autorizado | executa sob autorização prévia e reporta; só reversível | arquivar newsletters |

| Máximo nível 2 (nunca autónomas) | Podem chegar ao nível 3 |
|---|---|
| enviar email/mensagem | arquivar newsletters |
| apagar ficheiros ou dados | marcar como lido itens de baixo risco |
| cancelar reuniões | silenciar alertas duplicados |
| compras/pagamentos | outras operações claramente reversíveis e de baixo risco |
| qualquer ação destrutiva ou que vincule externamente | |

Regras: tudo começa no nível 1 ou 2; o nível 3 exige decisão explícita do Alexandre e
`max_autonomy_level` que o permita; o Echo pode **propor** subir o nível com base no histórico de
confirmações, mas **nunca sobe sozinho**.

### 8.2 Action Ledger

Registo de execução de primeira classe, não memória conversacional.

`ActionRecord`: `id`, `requested_by`, `source_channel`, `action_type`, parâmetros/referência,
`autonomy_level`, `confirmation_required`, `confirmation_result`, `executed_at`, `result`,
`reversible`, `undo_reference`, `status`.

"O que fizeste hoje?" responde-se **deterministicamente a partir do ledger**, nunca da memória.

## 9. Onde estamos (2026-11-11)

| Peça | Estado |
|---|---|
| Echo OS, voz (wake, VAD, barge-in, STOP, sleep), TTS Pocket | Funcional e estabilizado (ronda 1) |
| STT na GPU (faster-whisper turbo) | No main, opt-in; falta validação de qualidade e passar a default |
| Motor de conversa, tasks, workspace cognitivo | Funcional; conversa limitada pelo llama3.1:8b |
| Idioma, guards anti-invenção, pedidos vagos | Estabilizados |
| Memória longa | Limpa (55 memórias), sem embeddings; perfil pessoal quase vazio |
| Presença (estados, question discipline) | Fundação; política de atenção ainda não existe |
| Briefing / sugestões proativas | Esboço, só a partir de tarefas e memória local |
| Contratos do Core (EchoEvent, ledger, privacidade, autonomia) | **Não existem** |
| Núcleo separado da janela | **Não existe** |
| Agendador / vigias | **Não existem** (só o Context Observer dentro da app) |
| Email, calendário, sites, Telegram | **Não existem** |
| conversation.py ~15k linhas | Dívida estrutural (§10.1) |

## 10. Caminho

| Etapa | Objetivo | Critério de conclusão |
|---|---|---|
| **R2 — Fluidez da voz** (em curso) | Conversa por voz natural | STT GPU default; 1.º turno sem custos a frio (≤ ~0,5 s acima de um turno quente); mediana dos turnos ≈ 2,5 s; teste de voz real OK |
| **R3 — Cérebro e conhecimento** | Conversa deixa de ser "estranha" | Comparação de modelos com números; perfil-núcleo + contexto por relevância; decisão local/cloud aplicada |
| **F0 — Contratos do Core** | Definir os contratos antes de separar processos | EchoEvent, AttentionDecision, IncomingMessage/OutgoingNotification, ActionRequest/Result/Record, classes de privacidade, modelo de permissões/autonomia, API do Core e modelo de saúde com **schemas/modelos explícitos, testes e disciplina mínima de versões/compatibilidade**; Echo OS e vigias futuros dependem só dos contratos, nunca de implementações |
| **F1 — Core persistente** | Correr o Core como processo/serviço atrás desses contratos | **Bloqueada até à F0 concluída e ao requisito do §10.1.** Arranque automático e reinício; Echo OS liga-se ao Core; nada perde funcionalidade |
| **F2 — Primeiro vigia: sites** | Caminho completo vigia → evento → atenção → entrega | Site em baixo gera um aviso, uma vez, no canal certo; recuperação avisada; horas de silêncio respeitadas |
| **F3 — Telegram** | Falar com o Echo fora do PC | Mensagens e resumo diário; allowlist; `source_channel` preservado |
| **F4 — Calendário** | Saber o dia do Alexandre | Briefing da manhã com agenda real; avisos antes de reuniões |
| **F5 — Email (só leitura)** | Saber o que chegou e importa | Triagem local (`RESTRICTED`); resumo; nada enviado |
| **F6 — Ações com confirmação** | Responder, marcar, criar | Cada ação passa pela política de autonomia e fica no ledger |

Cada etapa fecha com testes e bateria verdes antes da seguinte; o modo estabilização continua a ser
a forma de trabalhar.

### 10.1 conversation.py

Reduzir de forma incremental, com testes e bateria verdes, sem reescrita gigante. Mas **a F1 fica
bloqueada até que as responsabilidades que atravessam a futura fronteira Core/UI tenham dono claro
num módulo próprio**, no mínimo: ciclo de vida da sessão, orquestração de tasks, orquestração de
ferramentas/ações, Presence, ciclo de vida da resposta e fronteira de coordenação voz/Core.
**A F1 não pode limitar-se a mover o monólito de conversation.py para outro processo.**

## 11. Decisões

Respondidas ficam registadas com data; abertas ficam marcadas **[aberta]**.

- (09/11) Idioma: setting fixo.
- (11/11) OpenJarvis/Ruflo: referência, não dependência.
- (11/11) Onde corre o Core: **no PC agora, híbrido depois** (§1.1).
- (11/11) Cérebro: **local + cloud com orçamento** (Ollama no dia a dia; Claude para conversa/triagem
  difícil, com limite diário), sujeito às classes de privacidade (§6).
- (11/11) Canal remoto: **Telegram**, bidirecional (§7).
- (11/11) Contas: **5 emails** — 2 pessoais, 1 do remo, 2 de trabalho (Leviahub, Devlop).
  **[aberta]** fornecedor de cada uma; autorização da empresa para as de trabalho.
- (11/11) Privacidade: **conversa e tarefas comuns podem ir ao Claude; conteúdo de emails não**,
  aplicado por classes atribuídas aos dados (§6).
- (11/11) Autonomia: **confirmação por omissão; autorização prévia configurável por tipo de ação**,
  com máximos por tipo (§8).
- (11/11) Sites — "estado" = **online**, **erros/certificado SSL/domínio**, **deploys/builds**.
  **[aberta]** lista de sites e plataformas de deploy — decidir em F2.
- (11/11) Idioma por omissão: **en-GB** (setting fixo).
- (11/11) Core: **monólito modular primeiro** (§2.1).
- **[aberta]** Horas de silêncio e classes com `override_quiet_hours` — decidir em F2.
- **[aberta]** Transporte da API/IPC do Core no PC (F1).
- **[aberta]** Anfitrião de longo prazo para 24/7 efetivo.
- **[aberta]** Detalhes da política de classificação de privacidade (incluindo regras de
  reclassificação de conteúdo derivado).
- **[aberta]** Política de confiança/autorização do Telegram para além da allowlist.
- (12/11) Cérebro da conversa: **conversa geral → Claude Haiku; llama3.1:8b local como reserva**
  (sem rede, erro ou mais de 4 s do Claude, orçamento esgotado, classe não permitida). Base: avaliação
  de 60 casos (`evals/results/conversation/r3_brains_*`): responde ao perguntado 90% contra 68%,
  not_rude 97% contra 89%, ~0,07 USD por 100 turnos, +0,6 s por turno.
- (12/11) Documentos do workspace: **Claude só quando o Alexandre pede explicitamente sobre esse
  documento**, no próprio turno (`PRIVATE` com marca de pedido explícito).
- (12/11) Emails (F5): **nunca na cloud** (`RESTRICTED`).
- (12/11) Chamadas pagas: ligam-se por definição (`allow_paid_calls` em `data/user_settings.json`,
  desligado por omissão), dentro do orçamento existente (0,25 USD/dia, 0,05 USD por chamada);
  `ECHO_ALLOW_PAID_MODEL_CALLS` é só um override.
- (12/11) Classes de privacidade (§6): `provider_allowed(privacy_class)` verificado na fronteira do
  router em todos os modos; um pedido sem classe é `PRIVATE`.
