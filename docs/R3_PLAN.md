# R3 — Cérebro e conhecimento

Data: 2026-11-12. Etapa R3 do North Star (§10). Objetivo: a conversa deixa de ser "estranha" e o
Echo passa a saber quem é o Alexandre. Medir antes de mudar: baseline em
`evals/results/conversation/r3_baseline_llama3.1-8b_run1|2` (llama3.1:8b).

## Decisões (12/11)
- Perfil-núcleo: rascunho escrito pelo Claude, revisto pelo Alexandre; ficheiro local
  `data/profile/core_profile.md` (fora do git; o repositório é público).
- Memória nova: o Echo **propõe e pede confirmação** ("Queres que guarde isto?"); nunca guarda às
  escondidas.
- Cérebro: comparar 1–2 modelos locais recentes que caibam na GPU ao lado do STT (~7 GB livres)
  e o Claude com orçamento, no mesmo conjunto de avaliação; decide-se pelos números.
- Personalidade: **estilo Jarvis britânico** — educado, ligeiramente formal, humor seco, "Sir"
  muito ocasional em en-GB. **O toque de máquina é desejado**, mas na voz de quem consulta o
  sistema: o Echo fala; o sistema por trás é que está online ("All systems are running smoothly").
  Com moderação, com variação e **só com estado real** (saúde dos componentes), nunca inventado.

- Cérebro (decidido a 12/11, passo 7): **conversa geral → Claude Haiku; llama3.1:8b local como
  reserva**. Comparação em `evals/results/conversation/r3_brains_*` (60 casos): Haiku responde ao
  perguntado 90% contra 68%, not_rude 97% contra 89%, ~0,07 USD por 100 turnos, +0,6 s por turno;
  qwen3.5:9b e qwen3.5:4b não superaram o llama3.1:8b. Documentos do workspace só vão ao Claude
  com pedido explícito sobre esse documento; emails nunca. Chamadas pagas por definição
  (`allow_paid_calls`), com o orçamento de 0,25 USD/dia.

## Baseline (llama3.1:8b, 35 casos, 2 corridas)
| Propriedade | Taxa |
|---|---|
| idioma, 2.ª pessoa | 100% |
| sem estado inventado | 97% (inflacionado: não apanha factos do mundo inventados) |
| não rude | 88–90% |
| sem tiques | 85–88% (en-GB 74–82%) |
| sem pergunta desnecessária | 92% |
| responde ao perguntado | 29–43% (só 7 turnos automáticos) |
Pior área: frases confusas — nunca pede para repetir; inventa reuniões, horas, jantares.

## Passos (por ordem; cada um com testes, eval e bateria verdes)
| # | Passo | Critério |
|---|---|---|
| 1 | **Afinar a avaliação**: detetar factos do mundo inventados; mais casos de "não percebi"; trocar "sem tiques" por "sem repetição" + "estado de sistema só se verificado" | Baseline refeita com as métricas corrigidas |
| 2 | **Frases confusas**: usar a confiança do STT (avg_logprob, no_speech) e sinais do texto para responder "Sorry, I didn't quite catch that — could you say it again?" sem LLM | 0 invenções nos casos confusos |
| 3 | **Perfil-núcleo no composer**: carregar `data/profile/core_profile.md` (sempre presente, curto); classe de privacidade PERSONAL | Respostas usam o perfil quando relevante; eval sem regressões |
| 4 | **Personalidade**: guia de estilo Jarvis no composer (EN e PT); o toque de sistema usa o estado de saúde real (Ollama, STT, TTS, rede) quando o Alexandre pergunta como está o Echo | Sem repetição da mesma frase; 0 estados de sistema inventados; not_rude ≥ 98% |
| 5 | **Limpar memória de ruído**: deixar de gravar links abertos como "preferências"; apagar os existentes (com backup) | Só memória útil |
| 6 | **Memória com confirmação**: detetar factos duradouros, propor, guardar só com "sim"; "o que sabes sobre mim?" e "esquece isso" deterministas | Testes de propor/confirmar/recusar/esquecer |
| 7 | **Comparação de cérebros**: mesmo eval com llama3.1:8b, 1–2 locais recentes e Claude (orçamento, só dados PERSONAL/PUBLIC) | Tabela qualidade/latência/custo; decisão registada **Decidido (12/11):** conversa geral no Claude Haiku, llama3.1:8b como reserva; ver North Star §11 |
| 8 | **Voz e conteúdo longo**: por voz, o Echo resume rascunhos/documentos longos em vez de os ler inteiros; o texto completo fica no ecrã | 1.º som < 2,5 s em respostas com rascunho |

Critério de fecho da R3: passos 1–8 feitos; eval com o cérebro escolhido melhor que a baseline em
todas as propriedades; teste de voz real do Alexandre aprovado.
