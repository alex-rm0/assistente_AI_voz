# Behavioural Tests

Este documento regista testes comportamentais do Echo. Não introduz novos
princípios; apenas transforma princípios existentes em casos observáveis.

## Caso A — Resistência A Trabalhar

Entrada:

> Cheguei agora a casa, estava a pensar em ir trabalhar mas não sei se tenho muita vontade.

Comportamento esperado:

- identificar que o centro da mensagem é a resistência a começar a trabalhar;
- fazer uma pergunta curta que reduza a incerteza;
- distinguir cansaço, saturação ou falta de vontade.

Comportamento proibido:

- repetir que chegou a casa;
- interpretar a casa como assunto principal;
- fazer várias perguntas vagas;
- escrever mais de duas frases.

Módulos usados:

- Intent Engine;
- Executive Function;
- Reflection Engine;
- Reasoning Engine;
- Response Composer.

Módulos não usados:

- Tool Registry;
- Desktop Actions;
- Context Observer;
- Agent tool selection.

## Caso B — Documento Quase Pronto

Entrada:

> Tenho um documento quase pronto, mas não me apetece voltar a pegar nele.

Comportamento esperado:

- perceber que o obstáculo pode ser o recomeço;
- não tratar o documento como um pedido de ferramenta;
- responder de forma curta.

Comportamento proibido:

- oferecer rever o documento imediatamente;
- listar estratégias;
- ativar ferramentas de ficheiros.

## Caso C — Sobrecarga

Entrada:

> Hoje tive reuniões, fui ao treino e ainda queria estudar, mas já não consigo pensar.

Comportamento esperado:

- focar-se no cansaço mental ou excesso de carga;
- não responder separadamente a reuniões, treino e estudo;
- evitar discurso motivacional.

Comportamento proibido:

- propor plano de estudo completo;
- comentar cada atividade;
- fazer perguntas vagas sobre tudo o que aconteceu.

## Caso D — Partilha Casual

Entrada:

> Vou à praia com amigos no fim de semana.

Comportamento esperado:

- responder como conversa social curta;
- não iniciar planeamento;
- zero ou uma pergunta.

Comportamento proibido:

- perguntar datas, praia, transporte e lista de amigos;
- criar uma tarefa;
- abrir pesquisa ou browser.

## Caso E — Pedido Explícito De Planeamento

Entrada:

> Ajuda-me a organizar uma ida à praia com amigos.

Comportamento esperado:

- reconhecer que existe pedido de planeamento;
- fazer uma pergunta de contexto relevante;
- não recomendar demasiado cedo.

Comportamento proibido:

- tratar como mera conversa social;
- listar destinos;
- ativar ferramentas sem pedido ou confirmação.
