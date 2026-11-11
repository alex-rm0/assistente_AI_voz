# Parallel Stabilization Checkpoint

Date: 2026-11-11. Branch: `stab/paralelo`.

This is a technical checkpoint, not a production release or a merge into the
main checkout. Known conversational failures remain open.

## Scope

- Opt-in faster-whisper CUDA STT through an isolated, persistent subprocess;
  the existing openai-whisper default remains available.
- Human-reviewed STT quality harness and integration instructions. Recordings,
  transcripts, benchmark environments and generated reports are not included.
- Response-language propagation to deterministic datetime, presence, task,
  context, model and voice status replies and the email workflow.
- Bounded personal-context guidance for conversational invitations, repeated
  service-question filtering, and guards against unsupported plans/tool claims.
- Machine-like social phrasing remains permitted, with guidance to vary it;
  this does not authorize claims that diagnostics or external actions ran.
- Email recipient review, separation of setup instructions from email content,
  canonical draft storage on the existing semantic object, and workspace export.
- Revision validation before draft replacement: rejected output preserves both
  content and revision. Explicit time constraints and equivalent 24-hour times
  are checked. A narrow, time-only correction can execute without the LLM.

## Validation

- Full suite: **1983 passed, 1 xfailed**.
- Focused conversation/routing validation: **403 passed**.
- `compileall` and `git diff --check`: passed.
- Real PySide6/QWebEngine/QWebChannel session with the production AssistantEngine
  and Ollama `llama3.1:8b`, using isolated temporary memory and workspace:
  create email, correct the meeting to 9:30 PM, request export, provide a filename.
  The saved UTF-8 text matched semantic object `writing-email-draft`, revision 2,
  after normalization of Windows CRLF line endings only.
- The Qt correction and export did not require additional LLM calls; the initial
  draft used the local model. No Anthropic calls were made.
- The Qt checkpoint test used typed input; it is not a new microphone/STT test.

## Open Issues

- Declarative conversation containing "o que tenho para fazer" can incorrectly
  route to TASK_COMMAND instead of remaining conversational.
- Short requests such as "Nao percebi" do not reliably explain the previous reply.
- Some invitations to converse route as general knowledge and can elicit
  unsupported offers about a schedule. The lexical guards are not comprehensive.
- Optional questions and conversational rhythm still need live evaluation.
- STT can misrecognize proper names; recipient review is not a transcription fix.
- Draft revision checks are conservative structural/time checks, not a guarantee
  of semantic correctness for every arbitrary editing instruction. Selection
  among multiple drafts needs further work; ambiguous cases must not be guessed.

Configuration, credentials, personal workspace files, databases, audio and
temporary Qt logs are excluded. The main checkout is not modified or promoted
by this checkpoint.
