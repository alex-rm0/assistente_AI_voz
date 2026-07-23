# Voice Control Security

Esta nota define regras para comandos de voz futuros no Echo.

## Regras

- API keys nunca devem ser ditadas, transcritas ou configuradas por conversa normal.
- Chaves de serviços externos devem usar armazenamento seguro, como Windows Credential Manager.
- A UI nunca deve receber o valor de uma chave depois de guardada; recebe apenas `api_key_configured=true|false`.
- Ações pagas exigem configuração prévia e autorização explícita.
- Ativar Claude automático deve pedir confirmação visual antes de ficar persistente.
- Ações destrutivas continuam a exigir confirmação explícita.
- Comandos de voz devem mostrar feedback visual do que foi entendido antes de executar ações sensíveis.
- A origem do comando deve ser registada como `voice`, `ui`, `keyboard` ou `api`.

## Princípio

Voz é uma forma de entrada, não uma exceção às regras de segurança.
