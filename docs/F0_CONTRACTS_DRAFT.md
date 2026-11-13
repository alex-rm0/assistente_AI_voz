# F0 — Rascunho dos contratos do Core

Data: 2026-11-12. Referência: `docs/ECHO_NORTH_STAR.md` §3, §4, §6, §8 e §10 (F0). **Rascunho**: os
contratos vivem em `assistant/core/contracts/` e **nada do runtime os importa**. Testes em
`tests/test_core_contracts.py`; JSON guardados da versão 1.0 em `tests/fixtures/core_contracts/v1/`.

## Mecânica comum

Cada contrato é uma `dataclass(frozen=True)` sobre a base `Contract` (`_base.py`), só com a
biblioteca padrão. Construir em código ou ler de JSON passa pelo mesmo caminho: os valores são
convertidos (strings de enum, timestamps ISO 8601, objetos aninhados) e depois correm os invariantes
do contrato. Os erros são `ContractError` (subclasse de `ValueError`) e começam pelo nome do campo.

- **Identidade e versão no JSON**: `to_dict()`/`to_json()` escrevem `contract` (o nome) e
  `contract_version` (`"MAJOR.MINOR"`, hoje `"1.0"` em todos). Um objeto construído em código está
  sempre na versão atual do seu leitor.
- **Compatibilidade**: o leitor aceita só o seu major. Campos desconhecidos são erro no minor do
  leitor ou anterior (são gralhas) e são ignorados vindos de um minor mais novo (campos aditivos e
  opcionais). Campos opcionais em falta ficam com o valor por omissão, por isso um 1.1 lê um 1.0.
  Mudança incompatível = novo major.
- **Imutabilidade**: mapas (`payload`, `parameters`, `metadata`, `output`) ficam congelados em
  profundidade (`MappingProxyType` e tuplos) e só aceitam valores JSON (sem `NaN`, sem objetos).
  São excluídos do `hash`.
- **Timestamps**: `datetime` com fuso obrigatório; sem fuso é erro.
- **Strings obrigatórias** não podem ser vazias. Nomes de tipo, fonte, canal e componente são
  identificadores minúsculos com pontos (`site.down`, `watcher.sites`, `telegram`,
  `provider.ollama`).

## Contratos

**PrivacyClass** (`privacy.py`). `PUBLIC < PERSONAL < PRIVATE < RESTRICTED`, atribuída aos dados.
`EchoEvent`, `IncomingMessage`, `OutgoingNotification` e `ActionRequest` têm `privacy_class`
obrigatória, sem valor por omissão: quem cria o objeto tem de decidir. `derived_privacy_class(*fontes)`
devolve a classe mais restritiva das fontes (conteúdo derivado herda). Reclassificar para baixo
não existe aqui: será uma regra explícita noutro sítio. `provider_allowed()` fica no router de
modelos, não no contrato. `ECHO_OS_DEFAULT_PRIVACY_CLASS = PERSONAL` é o valor que o adaptador do
Echo OS passa (resposta 1); o contrato continua a exigir a classe.

**EchoEvent** (`events.py`). Os campos do §3.1. `importance` é um enum fechado
(`low/normal/high/critical`) em vez de um número, para a política de atenção ter regras
testáveis; `confidence` é um número em [0, 1]. `status` é `active/resolved/superseded`
(a condição ainda se mantém / uma transição posterior fechou-a / substituído por um evento mais
novo com a mesma `dedupe_key`). **O evento não contém a decisão**: não tem esses campos e o leitor
rejeita `decision`, `attention` e `attention_decision` em qualquer versão, mesmo vindos de um minor
mais novo.

**AttentionDecision** (`events.py`). `id`, `event_id`, `decision`
(`INTERRUPT/DIGEST/RECORD_ONLY/DROP`), `decided_at`, `reason` obrigatória (a política tem de ser
explicável) e `deliver_after` opcional. As horas de silêncio mudam o momento, não a decisão:
um `INTERRUPT` diferido continua `INTERRUPT` com `deliver_after`; só o `INTERRUPT` pode ter
`deliver_after`, e nunca antes de `decided_at`.

**IncomingMessage** (`channels.py`). `id`, `source_channel`, `sender_id`, `text`, `received_at`,
`privacy_class`, `attachments` (tuplo de `AttachmentRef`: `id`, `media_type`, `reference`, `name`)
e `metadata` (mapa JSON livre do canal). Precisa de texto ou de pelo menos um anexo.

**OutgoingNotification** (`channels.py`). `id`, `target_channel`, `recipient`, `created_at`,
`priority` (`low/normal/high`), `privacy_class`, conteúdo (`text` e/ou `content_ref`, pelo menos um)
e duas ligações opcionais: `attention_decision_id` (a decisão que esta entrega executa) e
`in_reply_to_message_id` (resposta a uma `IncomingMessage`). Acrescentei `privacy_class`, que o
§3.3 não lista, porque o conteúdo de uma notificação é derivado e herda a classe (§6).
`content_free` marca um aviso sem conteúdo ("Chegou um email importante de X"). Para um canal fora
de `LOCAL_CHANNELS` (`echo_os`; hoje o Telegram é o único externo), uma notificação `PRIVATE` ou
`RESTRICTED` tem de ser `content_free` e não leva `content_ref` (resposta 2). O contrato não consegue
ler o texto: `content_free` é a declaração de quem cria o aviso, e a regra impede o resto.

**Modelo de autonomia: `ActionPolicy`** (`actions.py`). Por tipo de ação: `action_type`,
`autonomy_level`, `max_autonomy_level` (`AutonomyLevel`, `IntEnum` 0–3), `reversible`,
`external_effect`, `risk_class` (`low/medium/high`) e, só no nível 3, `authorized_by` e
`authorized_at`. Invariantes: nível ≤ máximo; máximo 3 só para ações reversíveis, de risco baixo e
fora de `NEVER_AUTONOMOUS_ACTION_TYPES` (enviar email/mensagem, apagar ficheiros/dados, cancelar
reuniões, pagamentos, compras e, desde 12/11, partilhar/publicar dados, mudar permissões/credenciais,
aceitar convites/compromissos: `data.share`, `data.publish`, `permissions.change`,
`credentials.change`, `calendar.accept_invitation`, `commitment.accept`); nível 3 exige autorização prévia registada (é autorização dada
antes, não bypass); `confirmation_required` é verdadeiro em todos os níveis menos o 3.
"Tudo começa no nível 1 ou 2" e "nunca sobe sozinho" são regras de processo, não de forma: ficam
para quem gravar as políticas (a autorização obrigatória no nível 3 é a parte verificável aqui).

**ActionRequest** (`actions.py`). `id`, `action_type`, `requested_by`, `source_channel`,
`requested_at`, `privacy_class`, `parameters` e uma cópia da `ActionPolicy` em vigor no momento do
pedido (o pedido fica auditável mesmo que a política mude depois). Níveis 0 e 1 nunca geram um
pedido executável; o `action_type` tem de bater com a política.

**ActionResult** (`actions.py`). `request_id`, `outcome` (`succeeded/failed`), `completed_at`,
`output`, `error`, `undo_reference`. Falha exige `error` e não tem `undo_reference`; sucesso não tem
`error`. Uma ação que nunca correu não tem resultado: isso é estado do registo.

**ActionRecord** (`actions.py`). Os campos do §8.2 mais `request_id` e `recorded_at`; `status` é
`awaiting_confirmation/not_executed/succeeded/failed` e `confirmation_result` é
`pending/confirmed/declined/expired/not_required`. Invariantes principais: só níveis 2 e 3 chegam
ao ledger; `confirmation_required` é verdadeiro exatamente no nível 2; **nunca executado sem
confirmação** quando é exigida; nível 3 só reversível; executado ⇔ tem `executed_at` e `result`, com
o mesmo `request_id` e o mesmo desfecho; `pending` ⇔ `awaiting_confirmation`; `undo_reference` só em
ações reversíveis com sucesso, igual ao do resultado. `confirmation_channel` diz de onde veio a
resposta: pode ser diferente do `source_channel` (pedido no PC, confirmado no Telegram), mas só um
canal de `CONFIRMATION_CHANNELS` (`echo_os`, `telegram`) responde; é obrigatório numa resposta
(`confirmed`/`declined`) e vazio sem resposta (resposta 3).

**HealthState / ComponentHealth** (`health.py`). `healthy/degraded/unavailable`, com os valores
em minúsculas como no §2.1.10. `ComponentHealth` tem `component`, `state`, `checked_at`, `reason`
(obrigatória fora de `healthy`) e `since` (não pode ser depois de `checked_at`). Não há agregação
"estado global do Core": falhar parcialmente significa mostrar cada componente.

## Fora deste rascunho

- **API do Core** (fronteira Echo OS ↔ Core, que substitui o `responder.__self__` + `getattr` do
  controller): depende do transporte da F1, que está **[aberta]** no North Star. Estes contratos são
  o vocabulário dessa API, não a API.
- Event Store, política de atenção, router de privacidade, executor e ledger: são implementações
  que vão consumir estes contratos.

## Alinhamento com `assistant/turn_input.py`

O `TurnInput` já foi desenhado a pensar nisto (a docstring diz-o) e encaixa sem o alterar. Um
adaptador futuro (`TurnInput` → `IncomingMessage`, num módulo novo do lado do Echo OS) faria:

| `TurnInput` | `IncomingMessage` |
|---|---|
| `message_id` | `id` |
| `text` | `text` |
| `source_channel` (`"echo_os"`) | `source_channel` |
| `received_at` (string ISO UTC) | `received_at` (convertido para `datetime`) |
| — | `sender_id`: o utilizador local (ex.: `"alexandre"`); o Echo OS só tem um |
| — | `privacy_class`: `ECHO_OS_DEFAULT_PRIVACY_CLASS` (`PERSONAL`, resposta 1) |
| `source` (`typed`/`voice`) | `metadata["modality"]` |
| `capture_id`, `session_id`, `capture_mode` | `metadata` |
| `stt_model`, `avg_logprob`, `no_speech_probability`, `compression_ratio` | `metadata` (só voz; `None` não é escrito) |

A regra do `TurnInput` "texto escrito não traz evidência acústica" passa a ser uma regra do
adaptador. Uma captura de voz com texto vazio não gera `IncomingMessage` (o contrato exige texto ou
anexo), o que coincide com o caminho atual de "não percebi". O fixture
`tests/fixtures/core_contracts/v1/IncomingMessage.json` mostra uma mensagem de voz com estes
metadados.

## Respostas do Alexandre (12/11)

1. **Mensagens do Echo OS** (janela/voz) são `PERSONAL` por omissão, até haver um classificador.
2. **Conteúdo `PRIVATE`/`RESTRICTED` nunca vai numa `OutgoingNotification` do Telegram**: só avisos
   sem conteúdo ("Chegou um email importante de X").
3. **Uma ação pode ser confirmada noutro canal**, desde que esse canal esteja na allowlist
   (`CONFIRMATION_CHANNELS`); canais desconhecidos nunca.
4. **Lista "nunca autónomas"**: confirmada, com mais partilhar/publicar dados, mudar
   permissões/credenciais e aceitar convites/compromissos.

Os contratos e os testes (`tests/test_core_contracts.py`, secção "Alexandre's answers of 12/11")
já seguem estas respostas. Os fixtures v1 de `ActionRecord` e `OutgoingNotification` ganharam os
campos novos (`confirmation_channel`, `content_free`) sem mudar de versão: é um rascunho que nada
lê ainda.

## Perguntas que estavam em aberto (respondidas acima)

1. **Classe por omissão das mensagens do Echo OS.** O contrato obriga a escolher. Até haver um
   classificador, uma mensagem escrita ou falada no PC entra como `PERSONAL` (pode ir ao Claude) ou
   como `PRIVATE` (só local salvo regra)? Bloqueia ligar o `TurnInput` aos contratos na F1.
2. **Conteúdo `PRIVATE`/`RESTRICTED` em notificações remotas.** O Telegram é um serviço externo. Uma
   `OutgoingNotification` derivada de um email (`RESTRICTED`) pode ir para o Telegram, só como
   referência ("tens 3 emails importantes") ou nunca? Bloqueia a F3/F5.
3. **Confirmação noutro canal.** Um pedido de nível 2 feito no PC pode ser confirmado pelo Telegram
   (e vice-versa)? Se sim, o `ActionRecord` precisa de `confirmation_channel`; se não, a regra fica na
   política de autorização. Bloqueia a F6.
4. **Lista "nunca autónomas".** Confirma os tipos em `NEVER_AUTONOMOUS_ACTION_TYPES` (enviar
   email/mensagem, apagar ficheiros/dados, cancelar reuniões, pagamentos, compras) e se há outros
   (ex.: aceitar convites de calendário, partilhar ficheiros).
