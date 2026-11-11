# Echo — North Star

Data: 2026-11-11. Este documento é a direção geral do projeto. Os outros docs (ROADMAP,
ECHO_ROADMAP, KNOWN_ISSUES, bateria) descrevem passos; este descreve **para onde vamos**.
Agentes (Claude Code, Codex): leiam isto antes de propor trabalho novo.

## 1. O destino

O Echo é uma **presença pessoal contínua** do Alexandre:

- **Está sempre ligado** (24/7), mesmo sem janela aberta.
- **Acompanha o mundo do Alexandre**, não só o PC: emails, calendário, tarefas,
  as páginas web que ele criou, projetos em curso.
- **Fala com ele onde ele estiver**: por voz e workspace no PC; por mensagem no telemóvel.
- **Interrompe pouco e bem**: avisa só o que importa, quando importa; o resto vai para resumos.
- **Conhece-o** (perfil e memória curados) e **nunca inventa** (grounding acima de fluidez).
- **Age com confirmação**: lê livremente o que lhe for autorizado; ações com efeito externo
  (enviar, marcar, apagar) pedem sempre confirmação.

Não é: um chatbot, um wrapper de LLM, um agente autónomo que faz coisas sozinho, nem uma
framework genérica (ver decisão sobre OpenJarvis/Ruflo: referências, não dependências).

## 2. Arquitetura alvo

```
┌──────────────────────────── ECHO CORE (serviço, sempre ligado) ────────────────────────────┐
│ Agendador ── Vigias (watchers): sites · calendário · email · tarefas · projetos (git)      │
│                 │ eventos normalizados                                                      │
│                 ▼                                                                           │
│ Triagem: regras determinísticas primeiro; LLM só para classificar/resumir                  │
│                 ▼                                                                           │
│ Política de atenção (Presence): agora / próximo resumo / silêncio  + horas de silêncio     │
│                 ▼                                                                           │
│ Memória & conhecimento: perfil curado · memória longa · tarefas · timeline · eventos       │
│ Motor de conversa (AssistantEngine) · Ferramentas (com permissões e confirmação)          │
│ Cérebro substituível: modelo local (Ollama) + modelo cloud opcional, por routing/orçamento │
└───────────────┬───────────────────────────────┬───────────────────────────────┬────────────┘
                │ API local                      │ canal remoto                  │ logs/saúde
                ▼                                ▼                               ▼
     Echo OS no PC (janela, voz,        Telemóvel (mensagens/push,       Painel de estado
     workspace, wake word, TTS)         pedidos curtos, resumos)          (o Echo está vivo?)
```

Princípios técnicos:

1. **Núcleo separado da interface.** A janela e a voz são clientes do Core; fechar a janela não
   desliga o Echo. (Hoje tudo vive dentro do processo Qt — é a mudança estrutural principal.)
2. **Vigias baratos e determinísticos.** Verificar um site, ler a agenda ou listar emails novos
   não precisa de LLM. O LLM só entra para decidir importância e escrever resumos.
3. **Um único sítio decide interromper** (camada de presença), com regras explícitas e testáveis.
4. **Credenciais nunca no código nem no git** (secret_storage), permissões mínimas (só leitura
   primeiro).
5. **Fiabilidade antes de capacidade**: reinício automático, logs, verificação de saúde, testes
   e bateria. Um assistente 24/7 que crasha às 3h não serve.
6. **Cérebro trocável**: tudo passa pelo ModelProvider; nenhuma lógica depende de um modelo.

## 3. Onde estamos (2026-11-11)

| Peça | Estado |
|---|---|
| Echo OS, voz (wake, VAD, barge-in, STOP, sleep), TTS Pocket | Funcional e estabilizado (ronda 1) |
| STT na GPU (faster-whisper turbo) | No main, opt-in; falta validação de qualidade e passar a default |
| Motor de conversa, tasks, workspace cognitivo | Funcional; conversa limitada pelo llama3.1:8b |
| Idioma, guards anti-invenção, pedidos vagos | Estabilizados |
| Memória longa | Limpa (55 memórias), sem embeddings; perfil pessoal quase vazio |
| Presença (estados, question discipline) | Fundação; decisão de interromper ainda não existe |
| Briefing / sugestões proativas | Esboço, só a partir de tarefas e memória local |
| Núcleo separado da janela | **Não existe** |
| Agendador / vigias | **Não existem** (só o Context Observer dentro da app) |
| Email, calendário, sites | **Não existem** |
| Canal no telemóvel | **Não existe** |
| conversation.py ~15k linhas | Dívida estrutural; reduzir antes de crescer muito mais |

## 4. Caminho (ordem e porquê)

| Etapa | Objetivo | Critério de conclusão |
|---|---|---|
| **R2 — Fluidez da voz** (em curso) | Conversa por voz natural | STT GPU default; 1.º som < 2,5 s; teste de voz real OK |
| **R3 — Cérebro e conhecimento** | Conversa deixa de ser "estranha" | Comparação de modelos com números; perfil pessoal curado sempre presente; decisão local/cloud |
| **F1 — Echo Core** | Echo vive sem janela | Serviço com arranque automático e reinício; Echo OS liga-se a ele; nada perde funcionalidade |
| **F2 — Primeiro vigia: sites** | Caminho completo vigia → atenção → aviso | Site em baixo gera aviso certo, uma vez, no canal certo; horas de silêncio respeitadas |
| **F3 — Canal remoto** | Falar com o Echo fora do PC | Mensagens e resumo diário no telemóvel |
| **F4 — Calendário** | Saber o dia do Alexandre | Briefing da manhã com agenda real; avisos antes de reuniões |
| **F5 — Email (só leitura)** | Saber o que chegou e importa | Triagem por importância; resumo; nada enviado sem confirmação |
| **F6 — Ações com confirmação** | Responder, marcar, criar | Cada ação externa pede sim/não e fica registada |
| Contínuo | Reduzir conversation.py | Por módulos, só com bateria verde |

Regra: cada etapa fecha com bateria e testes verdes antes da seguinte (o modo estabilização
continua a ser a forma de trabalhar, agora com um destino claro).

## 5. Decisões

Respondidas ficam registadas aqui com data; abertas ficam marcadas **[aberta]**.

- (09/11) Idioma: setting fixo.
- (11/11) OpenJarvis/Ruflo: referência, não dependência.
- (11/11) Onde corre o Core: **no PC agora, híbrido depois** (desenho preparado para mover os vigias para máquina pequena/cloud).
- (11/11) Cérebro: **local + cloud com orçamento** (Ollama no dia a dia; Claude para conversa/triagem difícil, com limite diário).
- (11/11) Canal remoto: **Telegram** (nos dois sentidos).
- (11/11) Contas: **5 emails** — 2 pessoais, 1 do remo, 2 de trabalho (Leviahub, Devlop).
  **[aberta]** fornecedor de cada uma; autorização da empresa para as de trabalho.
- (11/11) Privacidade: **conversa e tarefas podem ir ao Claude; conteúdo de emails não** — é lido e
  triado só pelo modelo local.
- (11/11) Autonomia: **avisa e sugere; age só com o sim do Alexandre**, com **autonomia configurável
  por tipo de ação**. Quando uma ação provar ser fiável, o Alexandre pode promovê-la (ver §6).
- (11/11) Sites — "estado" = **online**, **erros/certificado SSL/domínio**, **deploys/builds**.
  **[aberta]** lista de sites e plataformas de deploy (Railway, Vercel, GitHub Actions…) — decidir em F2.
- (11/11) Idioma por omissão: **en-GB** (setting fixo).
- **[aberta]** Horas de silêncio e o que pode interromper sempre — decidir em F2.

## 6. Autonomia configurável

Cada tipo de ação tem um nível, guardado em configuração e alterável pelo Alexandre:

| Nível | O Echo… | Exemplo |
|---|---|---|
| 0 — Desligado | não faz nem sugere | — |
| 1 — Avisa | só informa | "O site X está em baixo." |
| 2 — Sugere | prepara a ação e pede sim/não | rascunho de resposta a um email |
| 3 — Age e reporta | faz sozinho e diz o que fez; reversível | arquivar newsletters |

Regras: todas as ações começam no nível 1 ou 2; nível 3 só para ações reversíveis e só por decisão
explícita do Alexandre; cada ação fica num registo consultável ("o que fizeste hoje?"); o Echo pode
**propor** subir o nível de uma ação com base no histórico de confirmações, mas nunca sobe sozinho.
