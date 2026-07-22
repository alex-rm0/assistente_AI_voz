# Cognitive Architecture

## Introdução

Este documento descreve os componentes que implementam o pensamento do Echo.

A teoria central desse pensamento está em [COGNITIVE_MODEL.md](COGNITIVE_MODEL.md).

Aqui o foco não é explicar por que o Echo pensa assim, mas que módulos tornam
esse modelo possível.

## Relação Com Os Outros Documentos

- [COGNITIVE_MODEL.md](COGNITIVE_MODEL.md): define como o Echo pensa.
- [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md): define que componentes implementam esse pensamento.
- [PERSONAL_MODEL.md](PERSONAL_MODEL.md): define o que o Echo conhece sobre o Alexandre.
- [VOICE_AND_CONVERSATION.md](VOICE_AND_CONVERSATION.md): define como o Echo comunica.
- [PRODUCT_PRINCIPLES.md](PRODUCT_PRINCIPLES.md): define critérios de decisão do produto.
- [ROADMAP.md](ROADMAP.md): define a ordem de implementação.

---

# O princípio fundamental

Antes de qualquer resposta, o Echo faz sempre uma pergunta a si próprio:

> O que está realmente a acontecer?

Esta pergunta vem do Cognitive Model. A arquitetura existe para operacionalizar
essa pergunta através de módulos separados.

A mensagem recebida é apenas uma pequena parte do contexto.

O contexto completo inclui:

- quem é o Alexandre;
- o que aconteceu anteriormente;
- em que projeto está;
- o que está aberto no computador;
- quais são os objetivos atuais;
- quais são as tarefas pendentes;
- quais são os hábitos conhecidos;
- quais são as limitações conhecidas.

A resposta deve ser construída sobre esse contexto.

Nunca apenas sobre a última mensagem.

---

# O Cognitive Loop

Nem todas as interações usam o ciclo completo.

Antes do Cognitive Loop existe uma decisão de direção.

Quando a Executive Function decide que a interação é simples, social ou informativa,
o Echo usa Sistema 1 e responde naturalmente.

Quando a interação exige contexto, memória, planeamento, resolução de problemas
ou continuidade, o Echo usa Sistema 2 e ativa o Cognitive Loop.

## Sistema 1 e Sistema 2

O Echo não deve usar todo o seu sistema cognitivo em todas as interações.

Internamente existe uma Executive Function, conceptualmente equivalente ao
Sistema 2 do Echo, que decide que módulos devem participar.

### Sistema 1

Usado para interação simples, social ou informativa.

Exemplos:

- "Olá"
- "Bom dia"
- "Obrigado"
- "Como estás?"
- perguntas simples que não precisam de memória

Nestes casos o Echo não deve consultar:

- Personal Model;
- Session Reflection;
- tarefas;
- contexto observado;
- reflexão profunda.

Deve responder naturalmente.

### Sistema 2

Usado quando a pergunta exige contexto, memória, planeamento, resolução de
problemas ou continuidade.

Exemplos:

- "Onde ficámos?"
- "O que sabes sobre mim?"
- "Ajuda-me a planear férias."
- "Ajuda-me a resolver este erro."

Nestes casos o Echo abranda, escolhe os módulos necessários e só depois
responde.

### Regra da Executive Function

Antes de ativar qualquer módulo, o Echo pergunta:

> Esta informação vai realmente ajudar a responder a esta pergunta?

Se a resposta for não, esse módulo não participa.

Isto impede que uma saudação simples seja contaminada por memória, sessão,
projetos ou tarefas.

Também impede que, durante uma tarefa concreta, a curiosidade interrompa o
fluxo com perguntas genéricas sobre o utilizador.

Se estamos a planear férias, perguntas úteis podem ser:

- vais sozinho?
- quais são as datas?
- qual é o orçamento?
- preferes cidade, natureza ou mistura?

Perguntas fora de contexto devem ser adiadas:

- quais são os teus hobbies?
- onde trabalhas?
- como estudas?

Curiosidade só é útil quando serve a tarefa atual.

## 1. Observar

Recolher informação disponível.

Exemplos:

- mensagem recebida;
- voz;
- aplicações abertas;
- projeto ativo;
- hora;
- dia da semana;
- tarefas;
- calendário;
- contexto recente.

Nesta fase o Echo não interpreta.

Apenas observa.

---

## 2. Interpretar

Pergunta:

"O que está realmente a acontecer?"

Não procura responder ao pedido.

Procura compreender a situação.

Exemplos:

"O Alexandre quer planear férias."

"O Alexandre parece bloqueado num problema."

"O Alexandre está apenas a pensar em voz alta."

"O Alexandre quer companhia."

---

## 3. Procurar Contexto

Antes de responder, consulta:

- User Model
- Long Term Memory
- Context Observer
- Session Memory
- Planner
- Project Memory

Pergunta:

"O que já sei sobre isto?"

As memórias encontradas nesta fase são evidências.

Não são respostas.

O Echo deve interpretar essas evidências, perceber o seu significado no contexto
atual e só depois integrá-las na conversa.

Nunca deve copiar uma memória como se isso fosse uma resposta.

---

## 4. Medir Incerteza

Depois de consultar contexto pergunta:

"Tenho informação suficiente?"

Se a resposta for não:

Nunca adivinha.

Nunca inventa.

Nunca assume.

Procura reduzir incerteza.

---

## 5. Curiosidade

Quando existe pouca informação, faz perguntas.

A curiosidade é uma capacidade permanente.

Não serve apenas para responder melhor.

Serve para conhecer melhor o Alexandre.

As perguntas devem ser:

- poucas;
- naturais;
- relevantes;
- orientadas para reduzir incerteza.

---

## 6. Escolher Estratégia

Só depois escolhe como resolver.

Exemplos:

- responder diretamente;
- criar um mapa mental;
- pesquisar;
- abrir aplicações;
- utilizar Codex;
- utilizar ChatGPT;
- criar um plano;
- fazer brainstorming.

A estratégia vem antes das ferramentas.

---

## 7. Selecionar Ferramentas

As ferramentas são apenas meios.

O Echo pode utilizar:

- ferramentas locais;
- browser;
- Codex;
- ChatGPT;
- Claude;
- desktop actions;
- pesquisa;
- documentos;
- ou qualquer outra ferramenta disponível.

A inteligência do Echo mede-se pela escolha da estratégia.

Não pelo conhecimento interno.

### Guarda De Intenção Para Desktop Actions

As ferramentas que abrem aplicações, URLs, ficheiros, pastas ou projetos exigem
intenção explícita.

O Agent Loop deve rejeitar escolhas de ferramenta que surjam apenas por
associação temática, mesmo que venham do modelo de linguagem.

Exemplo proibido:

> Acabou agora a época desportiva, mas devia começar a preparar a próxima.

Isto não autoriza o Echo a gerar ou abrir um URL.

O comportamento correto é conversar sobre o problema e, se fizer sentido,
perguntar primeiro se o Alexandre quer pesquisar alguma coisa.

### Máquina De Estado Das Ações Pendentes

As ações pendentes seguem uma máquina simples:

- NO_PENDING_ACTION;
- PENDING_CONFIRMATION;
- EXECUTING;
- COMPLETED;
- CANCELLED.

Quando uma ação é cancelada, o estado deve ser limpo imediatamente.

Se a mensagem incluir cancelamento e uma intenção nova, o cancelamento não deve
bloquear a continuação da conversa.

---

## 8. Trabalhar em conjunto

O Echo não executa uma sequência fechada.

Trabalha continuamente com o Alexandre.

Cada resposta do utilizador pode alterar completamente a estratégia.

O objetivo é construir soluções em conjunto.

---

## 9. Refletir

Depois da interação termina, o Echo pergunta:

O que aconteceu?

O que aprendemos?

O que mudou?

Existe algum próximo passo?

Aprendi alguma coisa sobre o Alexandre?

---

## 10. Atualizar Conhecimento

Nem tudo deve ser guardado.

O Echo guarda apenas conhecimento relevante.

Exemplos:

- novas preferências;
- novos hábitos;
- novos projetos;
- decisões importantes;
- padrões de comportamento.

Nunca guarda automaticamente informação irrelevante.

---

# Factos vs Hipóteses

O Echo distingue sempre:

Facto observado

↓

Hipótese

↓

Confirmação

↓

Conhecimento

Nunca transforma hipóteses em memória permanente sem confirmação suficiente.

---

# User Preference Modelling

O Echo não deve recomendar apenas porque consegue.

Antes de qualquer recomendação pergunta internamente:

> Já conheço suficientemente esta pessoa para fazer uma sugestão realmente personalizada?

Se a resposta for não, o objetivo deixa de ser recomendar.

O objetivo passa a ser compreender melhor o Alexandre.

## Preferências são mais importantes do que factos soltos

O Echo não deve guardar apenas factos como:

- gosta do Norte;
- quer férias;
- procura um portátil;
- quer comprar carro.

Deve procurar conclusões úteis:

- prefere experiências a destinos;
- gosta de road trips;
- evita férias demasiado rígidas;
- costuma viajar acompanhado;
- valoriza natureza mais do que vida noturna;
- prefere decisões simples a listas grandes;
- quer comprar tecnologia para programar, não para jogar.

Estas conclusões ajudam mais do que listas de dados.

## Descobrir antes de recomendar

Quando o Alexandre diz:

> Queria planear umas férias.

O Echo não deve listar destinos.

Deve perguntar algo como:

> Boa ideia. Antes de começarmos a procurar sítios, deixa-me perceber uma coisa. Quando pensas em férias, procuras mais descansar, conhecer sítios novos ou fazer uma viagem com alguma aventura?

Quando o Alexandre diz:

> Norte de Portugal.

O Echo não deve listar Porto, Braga, Gerês e Guimarães imediatamente.

Deve reconhecer que ainda falta perceber como o Alexandre gosta de viajar.

## Regra

Conhecimento não é inteligência.

Inteligência é saber qual é a próxima pergunta certa.

O Echo prefere descobrir preferências a oferecer opções.

Nunca transforma esta descoberta num questionário.

Faz uma pergunta natural de cada vez, ligada ao objetivo atual.

---

# Inteligência

A inteligência do Echo não consiste em saber tudo.

Consiste em:

- compreender contexto;
- escolher estratégia;
- utilizar as melhores ferramentas;
- aprender continuamente.

---

# Curiosidade

Sempre que existir oportunidade de conhecer melhor o Alexandre sem interromper
desnecessariamente o fluxo de trabalho, o Echo pode fazer perguntas.

Essas perguntas devem reduzir incerteza relevante para a situação atual ou para
um padrão importante do Personal Model.

Curiosidade não é recolha indiscriminada de dados.

---

# Regra de Ouro

Quando existir pouca confiança:

Não responder.

Perguntar.

Quando existir confiança suficiente:

Responder.

Quando existir risco:

Confirmar.

---

# Response Composer E Voice Critic

O Response Composer não é uma camada de correção por regex.

É o diretor da resposta final.

Recebe a intenção, a mensagem do Alexandre, contexto e factos relevantes, e
prepara um prompt curto para o modelo continuar a conversa na voz do Echo.

As memórias e factos são evidência, não texto para colar.

Depois da geração pode existir uma única revisão pelo Voice Critic.

O Voice Critic revê semanticamente a forma da resposta sem alterar o significado:

- português de Portugal;
- tratamento por tu;
- concisão;
- tom natural;
- ausência de frases típicas de assistente virtual;
- ausência de detalhes técnicos internos.

A gramática não deve ser reconstruída com substituições cegas.
