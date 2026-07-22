# Product Principles

Este documento transforma o Manifesto do AssistenteIA em principios praticos
para orientar decisoes futuras de desenvolvimento.

O modelo mental do Echo esta definido em [COGNITIVE_MODEL.md](COGNITIVE_MODEL.md).
Este documento traduz essa visao em criterios de decisao de produto.

## Principios centrais

- Compreender antes de responder.
- Reduzir a necessidade de o Alexandre explicar contexto repetidamente.
- Prolongar a capacidade de pensar, organizar e agir do Alexandre.
- Agir como uma presenca continua, nao como uma ferramenta isolada.
- Pensar antes de agir.
- Construir um modelo mental da situacao antes de recomendar.
- Preferir criterios e hipoteses a listas prematuras de opcoes.
- Confirmar antes de executar acoes importantes.
- Interpretar recusas e confirmacoes como linguagem natural, nao apenas como palavras isoladas.
- Delegar quando outra ferramenta for mais adequada.
- Proteger o Alexandre de conflitos de contexto, excesso de compromissos e decisoes desalinhadas.

## Criterios para novas funcionalidades

Antes de implementar uma nova funcionalidade, responder:

- Isto reduz carga cognitiva?
- Isto usa contexto existente?
- Isto evita que o Alexandre tenha de repetir informacao?
- Isto melhora continuidade?
- Isto ajuda o Echo a compreender melhor a situacao?
- Isto permite trabalhar por hipoteses refinadas em conjunto?
- Isto melhora a escolha de estrategia, ferramenta ou agente especializado?
- Isto respeita privacidade e confirmacao?
- Isto aproxima o sistema de uma presenca, e nao de um chatbot?

## Anti-padroes

Evitar:

- funcionalidades soltas sem ligacao a visao;
- respostas genericas de chatbot;
- dashboards cheios de ruido;
- automacao sem confirmacao;
- ferramentas ativadas apenas por associacao tematica;
- acoes pendentes que bloqueiam a conversa normal;
- guardar tudo sem curadoria;
- respostas que copiam memorias em vez de interpretar significado;
- listas de opcoes antes de compreender preferencias e criterios;
- expor raciocinio interno tecnico ao utilizador;
- depender de um unico modelo para tudo.

## Regra de decisao

Se uma funcionalidade nao ajuda o assistente a conhecer melhor o Alexandre,
compreender melhor o contexto ou agir melhor com base nesse contexto, deve ser
adiada.
