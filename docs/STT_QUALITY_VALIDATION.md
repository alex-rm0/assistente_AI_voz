# Validação STT antes de alterar a app

Comparar openai-whisper medium CPU (baseline atual) e faster-whisper
large-v3-turbo CUDA FP16. A app e a sua configuração não são alteradas.
As duas variantes recebem os mesmos WAVs, idioma e initial_prompt.

## Recolha

No PowerShell, para os worktrees atuais:

```powershell
$appRoot = 'C:\Users\alexm\.vscode\projects\assistenteIA'
$parallelRoot = 'C:\Users\alexm\.vscode\projects\assistenteIA-par'
$dataset = "$appRoot\data\debug\voice_captures\stt_quality\dataset.json"
$python = "$appRoot\.venv\Scripts\python.exe"
$script = "$parallelRoot\scripts\stt_quality.py"
& $python $script prepare --dataset $dataset
& $python $script record --dataset $dataset
```

São 20 casos: conversa, nomes, lugares, números, datas, correções, comandos,
frases longas, pausas, voz baixa, ruído e dois silêncios. Usa ruído moderado
habitual nos casos `noise`. Nos silêncios, não fales. O microfone é o
predefinido do sistema. A captura só começa após Enter; espera por `A ouvir`.
Não são usados serviços de transcrição para criar a referência.

Pode gravar-se um único caso com `record --dataset $dataset --case stt_03`.
WAVs existentes nunca são substituídos. Para repetir toda a recolha, usa
um novo caminho de dataset.

## Referências revistas

O campo `prompt` é apenas a frase proposta. Ouve cada WAV e corrige
`reference` no JSON para refletir exatamente o que disseste, incluindo
hesitações ou correções. Só depois coloca `reviewed: true` nesse caso.
Nos silêncios, mantém `reference` vazio. O avaliador recusa datasets com
áudio em falta ou referências não revistas. Não uses a saída do modelo
como referência sem a ouvires e corrigires.

## Comparação

Mantém `llama3.1:8b` carregado antes de comparar e confirma com `ollama ps`
que está em GPU. Não executar outros benchmarks em paralelo.

```powershell
ollama ps
$env:HF_HOME = "$appRoot\.hf_cache"
& $python $script compare --dataset $dataset `
  --app-python "$appRoot\.venv\Scripts\python.exe" `
  --bench-python "$appRoot\.venv_stt_bench\Scripts\python.exe"
```

Cada variante carrega uma vez e faz uma passagem de aquecimento separada.
Os relatórios JSON/Markdown ficam numa nova pasta junto do dataset,
dentro da área de capturas já ignorada pelo Git. Contêm texto pessoal;
não os adicionar ao Git.

## Leitura dos resultados

- WER: substituições, omissões e inserções de palavras; menor é melhor.
- CER: erros de caracteres. São ignoradas caixa e pontuação, não acentos.
- Números escritos por extenso vs algarismos podem dar erro textual apesar
  de serem semanticamente equivalentes; rever esses casos manualmente.
- Silêncio: qualquer saída não vazia é assinalada como possível alucinação;
  não recebe um WER artificial de zero.
- Erros de execução ficam sem pontuação e fazem o comando falhar.
- Comparar tempos após aquecimento, carregamento e GPU antes/load/pico.
  A memória é da GPU inteira, aproximada sob WDDM, não por processo.
- A revisão humana deve verificar especialmente nomes, datas, números,
  negações e perdas das primeiras palavras.

Esta comparação mede STT bruto. Não inclui VAD, rejeição de silêncio,
filtros de repetição da app nem latência ponta a ponta no Qt. Uma amostra
por caso serve de triagem, não de prova estatística. Só decidir uma
integração depois de rever os resultados; não há troca automática de STT.
