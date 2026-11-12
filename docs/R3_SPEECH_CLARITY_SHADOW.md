# R3 passo 2 — modo sombra

2026-11-12. Implementação aprovada **apenas em modo sombra** com base no
[diagnóstico](R3_CONFUSING_SPEECH_DIAGNOSIS.md). Não é uma declaração de compreensão
semântica nem de resolução dos erros de conversação.

## Transporte

`assistant/turn_input.py::TurnInput` é imutável. Contém texto, modalidade (`voice`/`typed`),
modelo STT, avg_logprob, no_speech_probability, compression_ratio, capture_id, session_id,
capture_mode, message_id, received_at UTC e source_channel.

Cada worker cria a sua identidade antes de gravar. O transcritor reporta os valores dentro
do contexto local dessa captura; não se lê o último resultado de um objeto partilhado.
O caminho rápido de wake reporta o modelo tiny; um fallback reporta o modelo que devolveu
a transcrição. Warmup noutra thread não fornece evidência à captura.

O sinal interno Qt transporta o objeto, com receção enfileirada num slot declarado do
controller. O sinal público para a UI mantém texto. O request worker entrega o objeto
ao engine; a fronteira existente de erros mantém-se. Sessões e capturas antigas são
rejeitadas quando já existe outra captura dona da entrega.

O contexto do turno termina no `finally`, inclusive em erro. Um pedido escrito cria um
envelope novo sem confiança. MIC manual continua sem auto-send: a edição/revisão no input
seguida de envio é `typed`, não herda metadados da gravação anterior.

Compatibilidade com North Star §3.3: message_id/text/source_channel/received_at podem mapear
para o futuro IncomingMessage, e os dados acústicos para metadata. **Esse contrato não foi
implementado**; remetente, anexos, classes de privacidade e adapters continuam fora desta fase.

## Regra e proteções

`assistant/speech_clarity.py` não chama modelos nem altera tasks/memória.
Sinais fortes: ligação incompleta no fim, cadeia de pelo menos quatro palavras funcionais
sem conteúdo e certas ligações claramente quebradas entre pronome e infinitivo.
Texto citado e construções interrogativas PT usuais não contam como ruído.

O candidato a repetição exige dois sinais de texto, ou um mais avg_logprob < -0.9 ou
no_speech_probability > 0.5. Estes valores são experimentais, para observar/calibrar;
ausência de dados e NaN não são confiança baixa. Compression ratio é transportado, mas
não é um novo limiar: a rejeição anterior de repetição do STT permanece intacta.

Proteções reutilizam segurança, router rápido, intenções canónicas e o estado dos campos.
Wake/sleep, confirmações, operações reconhecidas, nomes de destinatários e valores
estruturados não são substituídos pela regra. Uma task ativa por si só não protege
todo e qualquer texto. As proteções não executam os comandos.

## Operação

Em `config/settings.json`:

```json
"speech_clarity": { "enforce": false }
```

Cada turno de voz aceite pelo engine escreve no terminal `[SPEECH_CLARITY]`, com
capture_id/sessão/modo, would_ask_repeat, signals, protected e motivos, enforce e confiança.
O texto cru não é duplicado nessa linha. Não há logging de clareza para pedidos escritos.
Wake/standby e revisão manual também produzem observação protegida no controller;
silêncio/rejeições continuam a rearmar sem criar um turno no engine.

Com `enforce=false`, nem resposta, prompt, rota, ferramentas nem memória são alterados
pela decisão. Não há chamadas extra a LLM/embeddings. O ramo para futura ativação explícita
está testado: devolve a frase fixa no response_language, sem LLM nem guardar o texto confuso
no histórico. Não o ativar antes de analisar falsos positivos das sessões reais.

Os testes cobrem concorrência, wake/turbo, barge-in, captura seguinte, warmup, cancelamento,
captura atrasada, travessia QThread/controller/engine, texto escrito após voz, exemplos reais
do diagnóstico que não devem disparar e exemplos sintéticos claramente quebrados.
