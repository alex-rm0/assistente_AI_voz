from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


PORTUGUESE_PT = "pt-PT"
ENGLISH_GB = "en-GB"
SUPPORTED_RESPONSE_LANGUAGES = {PORTUGUESE_PT, ENGLISH_GB}


@dataclass(frozen=True)
class ResponseLanguageCommand:
    handled: bool
    language: str = ""
    confirmation: str = ""


@dataclass(frozen=True)
class TranslatePreviousResponseCommand:
    """Section 2/3 (Phase 7.3.4): a CONTEXTUAL, one-off request to
    transform/translate the PREVIOUS assistant reply into a target
    language -- distinct from ResponseLanguageCommand, which changes the
    PERSISTENT response_language preference going forward. Handling this
    must never call _set_response_language."""

    handled: bool
    target_language: str = ""


def normalize_response_language(value: object) -> str:
    text = str(value or "").strip()
    if text in SUPPORTED_RESPONSE_LANGUAGES:
        return text
    clean = _normalize(text)
    if clean in {"en", "english", "en-gb", "en gb", "ingles", "inglês"}:
        return ENGLISH_GB
    if clean in {"pt", "pt-pt", "pt pt", "portugues", "português", "portuguese", "portuguese portugal"}:
        return PORTUGUESE_PT
    return ""


def legacy_language_value(language: str) -> str:
    return "en" if normalize_response_language(language) == ENGLISH_GB else PORTUGUESE_PT


def response_language_instruction(language: str) -> str:
    normalized = normalize_response_language(language) or PORTUGUESE_PT
    if normalized == ENGLISH_GB:
        return (
            "High-priority response language:\n"
            "- Respond in English.\n"
            "- Prefer natural British English spelling and phrasing.\n"
            "- Do not mix Portuguese into the answer unless the user explicitly asks for it.\n"
            "- If the user asks to switch back to Portuguese, obey that language command immediately."
        )
    return (
        "Preferências de idioma:\n"
        "- idioma_base = pt-PT.\n"
        "- response_language = pt-PT.\n"
        "- Responde em português de Portugal.\n"
        "- Trata o Alexandre por tu.\n"
        "- Vocabulário pt-PT obrigatório quando responderes em português: "
        "aplicações, ecrã, acompanhar/observar, ficheiros, aceder.\n"
        "- Evita português do Brasil: aplicativos, tela, assistindo, arquivos, acessar."
    )


def response_language_status(language: str) -> str:
    return "English" if normalize_response_language(language) == ENGLISH_GB else "Português"


def detect_response_language_command(message: str) -> ResponseLanguageCommand:
    text = _normalize(message).strip(" .,!?:;")
    if not text:
        return ResponseLanguageCommand(False)
    if _asks_for_english(text):
        return ResponseLanguageCommand(True, ENGLISH_GB, "Sure. I'll respond in English.")
    if _asks_for_portuguese(text):
        return ResponseLanguageCommand(True, PORTUGUESE_PT, "Claro. Vou responder em português.")
    return ResponseLanguageCommand(False)


# Section 2/3 (Phase 7.3.4): phrases that refer DEICTICALLY to the
# assistant's own PREVIOUS reply ("essa resposta", "isso", "that") and ask
# for it in another language -- a one-off content transformation, never a
# persistent preference change. Checked (and consumed) BEFORE
# detect_response_language_command so a phrase like "quero essa resposta em
# inglês" -- which also contains the substring "resposta em ingles" that
# _asks_for_english matches on -- is never mistaken for a persistent
# language switch.
_TRANSLATE_PREVIOUS_TRIGGER_PHRASES = (
    "traduz essa resposta",
    "traduz esta resposta",
    "traduz isso",
    "traduz isto",
    "traduzir essa resposta",
    "quero essa resposta em",
    "quero essa resposta traduzida",
    "quero esta resposta em",
    "diz isso em",
    "diz isto em",
    "diz-me isso em",
    "diz me isso em",
    "repete isso em",
    "repete essa resposta em",
    "agora em ingles",
    "agora em portugues",
    "say that in english",
    "say that in portuguese",
    "say this in english",
    "say this in portuguese",
    "translate that",
    "translate this",
    "translate the previous",
)


def detect_translate_previous_response_command(message: str) -> TranslatePreviousResponseCommand:
    text = _normalize(message).strip(" .,!?:;")
    if not text:
        return TranslatePreviousResponseCommand(False)
    if not any(phrase in text for phrase in _TRANSLATE_PREVIOUS_TRIGGER_PHRASES):
        return TranslatePreviousResponseCommand(False)
    if "ingles" in text or "english" in text:
        return TranslatePreviousResponseCommand(True, ENGLISH_GB)
    if "portugues" in text or "portuguese" in text:
        return TranslatePreviousResponseCommand(True, PORTUGUESE_PT)
    # A trigger phrase matched (e.g. "translate that") but no explicit
    # target language was named -- never guess; let the caller fall through
    # to normal handling instead of silently picking a language.
    return TranslatePreviousResponseCommand(False)


def translation_instruction(target_language: str) -> str:
    normalized = normalize_response_language(target_language) or PORTUGUESE_PT
    label = "English" if normalized == ENGLISH_GB else "português de Portugal"
    return (
        f"Traduz o texto seguinte para {label}, mantendo o significado, o tom e o nível de detalhe. "
        "Devolve apenas o texto traduzido, sem comentários adicionais nem explicações sobre a tradução."
    )


def simple_social_response(message: str, language: str) -> str | None:
    text = _normalize(message).strip(" .,!?:;")
    text = re.sub(r"[,.!?:;]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return None
    english = normalize_response_language(language) == ENGLISH_GB
    if text in {"ola", "olá", "hello", "hi", "hey"}:
        return "Hi. How are you?" if english else "Olá! Como estás?"
    if text in {"viva tudo bem contigo", "ola tudo bem contigo", "olá tudo bem contigo", "tudo bem contigo"}:
        return "All good. How about you?" if english else "Tudo bem. E contigo?"
    if text in {
        "ola como estas",
        "olá como estás",
        "como estas",
        "como estás",
        "tudo bem",
        "how are you",
        "how are you doing",
    }:
        return "I'm good. How about you?" if english else "Estou bem. E tu?"
    if text in {"tambem estou bem", "também estou bem", "tambem estou bem obrigado", "também estou bem obrigado"}:
        return "Good to hear." if english else "Ainda bem."
    if text in {"bom dia", "good morning"}:
        return "Good morning." if english else "Bom dia."
    if text in {"boa tarde", "good afternoon"}:
        return "Good afternoon." if english else "Boa tarde."
    if text in {"boa noite", "good evening", "good night"}:
        return "Good evening." if english else "Boa noite."
    if text in {"obrigado", "obrigada", "muito obrigado", "muito obrigada", "thanks", "thank you"}:
        return "You're welcome." if english else "De nada."
    if text in {"ok obrigado", "ok obrigada", "esta bem obrigado", "está bem obrigado"}:
        return "You're welcome." if english else "De nada."
    if text in {"estas ai", "estás ai", "estas aí", "estás aí", "are you there"}:
        return "Yes, I'm here." if english else "Sim, estou aqui."
    if text in {"como te chamas", "qual e o teu nome", "qual é o teu nome", "what is your name"}:
        return "I'm Echo." if english else "Chamo-me Echo."
    return None


def looks_mixed_for_language(response: str, language: str) -> bool:
    text = _normalize(response)
    if not text:
        return False
    if normalize_response_language(language) == ENGLISH_GB:
        pt_markers = ("estou ", "estas ", "estás ", "contigo", "portugues", "ficheiro", "aplicacoes", "ecrã", "teu ")
        en_markers = (" i ", " you ", " the ", " sure ", " how ", " english ")
        return sum(marker in f" {text} " for marker in pt_markers) >= 2 and not any(marker in f" {text} " for marker in en_markers)
    en_markers = (" i'll ", " sure ", " how about you", " i can ", " english ")
    return any(marker in f" {text} " for marker in en_markers) and any(marker in text for marker in ("estou", "vou", "posso", "contigo"))


# Phase 7.7 (section 17): ONE small, explicit localization boundary for
# Echo-authored presentation strings (tool-result headers, task-artifact
# labels, deterministic fallbacks) -- deliberately not a general i18n
# framework, since only pt-PT/en-GB are in scope (section 17). Each entry
# is (portuguese, english); callers never branch on response_language
# themselves for a string that lives here, they just call localize(key, ...).
# This does NOT replace localize_deterministic_response above (which
# post-processes a FULL, already-composed reply against fixed sentence
# shapes) -- it's for individual presentation strings assembled by callers
# BEFORE composition (tool observations, task-artifact fields), which never
# reach that post-processing step at all (see the Phase 7.7 report).
_STRINGS: dict[str, tuple[str, str]] = {
    # --- assistant/tools.py: context-observer tool result headers -------
    "tool.windows_detected_header": ("Janelas detetadas:", "Open windows detected:"),
    "tool.no_system_context": (
        "Consigo tentar observar o computador, mas ainda não tenho dados suficientes. "
        "Experimenta mudar de janela ou aguardar alguns segundos.",
        "I can try to observe the computer, but I don't have enough data yet. "
        "Try switching windows or waiting a few seconds.",
    ),
    "tool.active_window": ("Janela ativa detetada: {name}", "Active window detected: {name}"),
    "tool.active_application": ("Aplicação ativa detetada: {name}", "Active application detected: {name}"),
    "tool.recent_activity_header": ("Atividade recente detetada:", "Recent activity detected:"),
    "tool.recent_activity_minutes": ("cerca de {minutes} min", "about {minutes} min"),
    "tool.recent_activity_header_with_summary": ("Atividade recente: {summary}", "Recent activity: {summary}"),
    "tool.active_marker": (" (ativa)", " (active)"),
    # --- assistant/task_intelligence.py: cognitive task view sections ---
    "task.section.decisions": ("Decisões", "Decisions"),
    "task.section.known": ("O que já sabemos", "What we already know"),
    "task.section.now": ("Neste momento", "Right now"),
    "task.section.organizing": ("A organizar", "Organising"),
    "task.item.accommodation": ("Alojamento", "Accommodation"),
    "task.item.transport": ("Transportes", "Transport"),
    "task.item.itinerary": ("Itinerário", "Itinerary"),
    "task.item.costs": ("Custos", "Costs"),
    "task.section.attention": ("Atenção", "Attention"),
    "task.section.still_missing": ("Ainda falta", "Still missing"),
    "task.section.next_step": ("Próximo passo", "Next step"),
    "task.section.options": ("Opções", "Options"),
    "task.section.found": ("Encontrado", "Found"),
    "task.section.limits": ("Limites", "Limits"),
    "task.section.priorities": ("Prioridades", "Priorities"),
    "task.recommendation.seems_best": ("{recommendation} parece a melhor opção", "{recommendation} looks like the best option"),
    "task.overview.comparison": ("Comparação", "Comparison"),
    "task.overview.planning": ("Planeamento", "Planning"),
    "task.overview.goal": ("Objetivo", "Goal"),
    # --- assistant/conversation.py: generic task title/card fallbacks ---
    "task.title.vacation_planning": ("Planeamento de férias", "Holiday planning"),
    "task.title.decision": ("Decisão em preparação", "Decision in progress"),
    "task.title.study_plan": ("Plano de estudo", "Study plan"),
    "task.title.research": ("Investigação", "Research"),
    "task.title.comparison": ("Comparação", "Comparison"),
    "task.title.comparison_category": ("Comparação: {category}", "Comparison: {category}"),
    "task.title.activity": ("Atividade em curso", "Activity in progress"),
    "task.title.writing": ("Escrita em conjunto", "Collaborative writing"),
    "task.title.generic": ("Tarefa", "Task"),
    "task.card.goal": ("Objetivo:", "Goal:"),
    # Built-in task goals (the stored ActiveTaskContext.goal values in
    # conversation.py/_TASK_METADATA); shown through localize_task_goal().
    "task.goal.compose_email": ("escrever um email", "write an email"),
    "task.goal.writing": ("escrever em conjunto", "write together"),
    "task.goal.plan_trip": ("planear férias", "plan a holiday"),
    "task.goal.make_decision": ("tomar uma decisão", "make a decision"),
    "task.goal.study_topic": ("estudar", "study"),
    "task.goal.research_topic": ("investigar opções", "research options"),
    "task.goal.generic_comparison": ("comparar opções", "compare options"),
    "task.goal.generic_activity": ("trabalhar num objetivo", "work towards a goal"),
    "task.goal.preference_discovery": (
        "continuar a compreender preferências antes de recomendar",
        "keep understanding preferences before recommending",
    ),
    "task.card.destination": ("Destino:", "Destination:"),
    "task.card.preference": ("Preferência:", "Preference:"),
    "task.card.theme": ("Tema:", "Theme:"),
    "task.card.category": ("Categoria:", "Category:"),
    "task.card.options": ("Opções:", "Options:"),
    "task.card.criteria": ("Critérios:", "Criteria:"),
    "task.card.requests": ("Pedidos:", "Requests:"),
    "task.card.notes": ("Notas:", "Notes:"),
    "task.card.collecting_context": ("[a recolher contexto]", "[gathering context]"),
    # --- assistant/task_formatter.py + long_term_memory.py: reminders ---
    # Task card subtitle: readable label for ActiveTaskContext.status (the
    # internal value itself is never changed).
    "task.card_status.collecting": ("a recolher informação", "collecting details"),
    "task.card_status.ready": ("pronta", "ready"),
    "task.card_status.working": ("a preparar", "working"),
    "task.card_status.reviewing": ("em revisão", "in review"),
    "task.card_status.paused": ("em pausa", "paused"),
    # compose_email task card.
    "email.card.title": ("Email", "Email"),
    "email.card.title_to": ("Email para {who}", "Email to {who}"),
    "email.card.role_article": ("o {role}", "the {role}"),
    "email.card.to": ("Para:", "To:"),
    "email.card.context": ("Contexto:", "Context:"),
    "email.card.purpose": ("Objetivo:", "Purpose:"),
    "email.card.subject": ("Assunto:", "Subject:"),
    "email.card.preparing": ("[conteúdo ainda em preparação]", "[content still being prepared]"),
    "email.card.ready": ("[pronto para redigir]", "[ready to draft]"),
    "email.role.professor": ("professor", "professor"),
    "email.role.colleague": ("colega", "colleague"),
    "email.role.client": ("cliente", "client"),
    "email.role.sponsor": ("patrocinador", "sponsor"),
    "email.role.manager": ("chefe", "manager"),
    "email.role.advisor": ("orientador", "supervisor"),
    "email.purpose.request_clarification": ("pedir esclarecimentos", "ask for clarification"),
    "email.purpose.request_help": ("pedir ajuda", "ask for help"),
    "email.purpose.schedule_meeting": ("marcar uma reunião", "schedule a meeting"),
    "email.purpose.thanks": ("agradecer", "say thank you"),
    "email.purpose.ask_exam_date": ("perguntar a data do exame", "ask for the exam date"),
    "task.status.pending": ("pendente", "pending"),
    "task.status.completed": ("concluída", "completed"),
    "task.status.cancelled": ("cancelada", "cancelled"),
    "task.panel.today_header": ("Para esse dia, tens", "For that day, you have"),
    "task.panel.for_today_header": ("Para hoje, tens", "For today, you have"),
    "task.panel.overdue_header": ("Tens em atraso", "You have overdue"),
    "task.panel.week_header": ("Esta semana, tens", "This week, you have"),
    "task.panel.pending_header": ("Tens pendente", "You have pending"),
    "task.panel.empty": ("Sem tarefas pendentes.", "No pending tasks."),
    "task.panel.no_tasks_for_date": (
        "Não tens tarefas registadas para {date}.",
        "You don't have any tasks recorded for {date}.",
    ),
    "task.panel.no_overdue": ("Não tens tarefas atrasadas.", "You don't have any overdue tasks."),
    "task.panel.no_tasks_in_range": (
        "Não tens tarefas registadas entre {start} e {end}.",
        "You don't have any tasks recorded between {start} and {end}.",
    ),
    "task.panel.no_pending_registered": (
        "Não tens tarefas pendentes registadas.",
        "You don't have any pending tasks recorded.",
    ),
    "task.action.cannot_query": ("Não consegui consultar as tarefas pendentes.", "I couldn't check the pending tasks."),
    "task.action.missing_manager": (
        "Não consegui alterar a tarefa. Ela continua pendente.",
        "I couldn't change the task. It's still pending.",
    ),
    "task.action.ambiguous_complete": (
        "Tenho várias tarefas pendentes. Diz-me qual queres marcar como concluída.",
        "I have several pending tasks. Tell me which one you want marked as completed.",
    ),
    "task.action.ambiguous_cancel": (
        "Tenho várias tarefas pendentes. Diz-me qual queres cancelar.",
        "I have several pending tasks. Tell me which one you want to cancel.",
    ),
    "task.action.ambiguous_postpone": (
        "Tenho várias tarefas pendentes. Diz-me qual queres adiar.",
        "I have several pending tasks. Tell me which one you want to postpone.",
    ),
    "task.action.completed": (
        "Está feito. Marquei essa tarefa como concluída e já não aparece nas pendentes.",
        "Done. I've marked that task as completed and it no longer shows up as pending.",
    ),
    "task.action.cancelled": (
        "Está feito. Cancelei essa tarefa e já não aparece nas pendentes.",
        "Done. I've cancelled that task and it no longer shows up as pending.",
    ),
    "task.action.no_task_to_postpone": (
        "Não encontrei uma tarefa pendente para adiar.",
        "I couldn't find a pending task to postpone.",
    ),
    "task.action.postponed": ("Adiei a tarefa '{title}' para {date}.", "I've postponed the task '{title}' to {date}."),
    "task.reminder.project_framing": (
        "uma tarefa relacionada com o projeto {project}: {description}",
        "a task related to the {project} project: {description}",
    ),
    "task.reminder.count_heading": ("{heading} {count} tarefas:", "{heading} {count} tasks:"),
    "task.reminder.detail.description": ("descrição:", "description:"),
    "task.reminder.detail.date": ("data:", "date:"),
    "task.reminder.detail.project": ("projeto:", "project:"),
    "task.reminder.detail.priority": ("prioridade:", "priority:"),
    "task.reminder.detail.status": ("estado:", "status:"),
    # --- assistant/response_composer.py: composer-owned fallback text ---
    # Final-response guards (AssistantEngine._complete_turn): Echo-authored
    # replacements for a blocked reply. No internal component names.
    "guard.tool_claim": (
        "Ainda não consegui fazer essa pesquisa — não tenho essa ferramenta ligada a esta conversa.",
        "I still can't do that research. I don't have that tool connected to this conversation.",
    ),
    "guard.memory_recall_claim": (
        "Não tenho essa informação guardada sobre isso.",
        "I don't have that information saved.",
    ),
    "guard.memory_claim": (
        "Não quero inventar memória. Posso consultar a memória real se quiseres.",
        "I don't want to make up a memory. I can check the real memory if you'd like.",
    ),
    "guard.historical_duration_claim": (
        "Não tenho memória suficiente para confirmar essa duração.",
        "I don't have enough memory to confirm that duration.",
    ),
    "guard.activity_claim": (
        "Não sei o que estiveste a fazer. Posso dizer-te que janela tens aberta agora, se me perguntares.",
        "I don't know what you were doing. I can tell you which window you have open right now if you ask me.",
    ),
    "guard.workspace_claim": (
        "Não alterei nada no workspace. Preciso de identificar um objeto real antes de agir.",
        "I didn't change anything in the workspace. I need to identify a real object before acting.",
    ),
    "composer.no_context_fallback": (
        "Tenho algum contexto sobre isso, mas ainda não o suficiente para te responder bem.",
        "I have some context about this, but not enough yet to give you a good answer.",
    ),
    "composer.local_safe_fallback": (
        "Lamento, algo correu mal ao preparar essa resposta. Podes tentar outra vez?",
        "Sorry, something went wrong preparing that reply. Could you try again?",
    ),
    "composer.local_safe_fallback.negative_result": (
        "Lamento que tenha corrido mal.",
        "I'm sorry that didn't go well.",
    ),
    "composer.local_safe_fallback.unmotivated": (
        "Percebo que isso te tenha deixado desmotivado.",
        "I understand that's left you feeling unmotivated.",
    ),
    "composer.local_safe_fallback.afraid": (
        "Percebo que isso te esteja a pesar.",
        "I understand that's weighing on you.",
    ),
    "composer.local_safe_fallback.sad": (
        "Lamento. Deve ser difícil.",
        "I'm sorry. That must be hard.",
    ),
    "composer.regeneration_instruction": (
        "A resposta anterior ofereceu ajuda em vez de executar um pedido de escrita completo.\n"
        "Executa agora o pedido diretamente. Não perguntes se o Alexandre quer ajuda.",
        "The previous reply offered help instead of carrying out a complete writing request.\n"
        "Carry out the request directly now. Don't ask whether Alexandre wants help.",
    ),
    # Phase 8.1 (Presence question discipline): only ever appended when
    # PresenceDecision.question_needed is False for the current turn --
    # composition guidance, not text post-processing (see
    # assistant/response_composer.py's _question_discipline_instruction).
    "composer.question_discipline.suppress_optional_followup": (
        "Responde diretamente ao que foi dito. Não acrescentes uma pergunta de seguimento só para "
        "manter a conversa. Só faças uma pergunta se for mesmo necessária para responder corretamente "
        "ou para avançar uma tarefa em curso.",
        "Answer directly. Do not add a follow-up question merely to keep the conversation going. "
        "Only ask a question if it is genuinely required to answer correctly or to progress an "
        "active task.",
    ),
    # --- prototype_web_ui/controller.py: TTS availability ----------------
    "tts.unavailable": (
        "Não encontrei uma voz TTS local compatível.",
        "I couldn't find a compatible local TTS voice.",
    ),
    # --- Phase 7.7.1: task_intelligence.py free-text dialogue/query -------
    # answers -- canonical values (subject/predicate/value, task_type,
    # destination, options, etc.) are NEVER translated here; only the
    # Echo-authored sentence scaffolding around them is.
    "task.join.and": ("e", "and"),
    "task.entry.budget_max": ("limite de orçamento: {value} euros", "budget limit: €{value}"),
    "task.entry.price": ("{subject} custa {value} euros", "{subject} costs €{value}"),
    "task.entry.camera_priority": ("a câmara pesa mais na decisão", "the camera matters more in the decision"),
    "task.entry.battery_priority.low": ("a bateria pesa menos na decisão", "the battery matters less in the decision"),
    "task.entry.battery_priority.medium": ("a bateria fica como fator secundário", "the battery is a secondary factor"),
    "task.entry.battery_priority.high": ("a bateria pesa bastante na decisão", "the battery matters a lot in the decision"),
    "task.entry.camera_quality": ("{subject} parece melhor em câmara", "{subject} looks better on camera"),
    "task.entry.battery": ("{subject} parece melhor em bateria", "{subject} looks better on battery"),
    "task.entry.support_priority": ("o suporte pesa bastante na decisão", "support matters a lot in the decision"),
    "task.entry.price_priority.low": ("o preço pesa menos na decisão", "price matters less in the decision"),
    "task.entry.price_priority.high": ("o preço pesa bastante na decisão", "price matters a lot in the decision"),
    "task.entry.location_priority": ("a localização pesa bastante na decisão", "location matters a lot in the decision"),
    "task.entry.support_quality.better": ("{subject} tem melhor suporte", "{subject} has better support"),
    "task.entry.support_quality.limited": ("{subject} tem suporte limitado", "{subject} has limited support"),
    "task.entry.support_quality.other": ("{subject}: suporte {value}", "{subject}: support {value}"),
    "task.entry.destination": ("destino: {value}", "destination: {value}"),
    "task.entry.origin": ("partida: {value}", "departure: {value}"),
    "task.entry.date_range": ("datas: {value}", "dates: {value}"),
    "task.entry.duration": ("duração: {value}", "duration: {value}"),
    "task.entry.travel_style": ("preferência: {value}", "preference: {value}"),
    "task.entry.activities": ("{value} em {subject}", "{value} in {subject}"),
    "task.next_step.close": ("Fechar {gaps}", "Close {gaps}"),
    "task.next_step.organize_trip": (
        "Organizar alojamento, transportes, itinerário e custos",
        "Organise accommodation, transport, itinerary and costs",
    ),
    "task.next_step.confirm_before_deciding": (
        "Confirmar os pontos em aberto antes de decidir",
        "Confirm the open points before deciding",
    ),
    "task.travel.great_have_destination": ("Ótimo. Já temos {destination}", "Great. We already have {destination}"),
    "task.travel.and_dates": (" e as datas, {date_range}", " and the dates, {date_range}"),
    "task.travel.now_organize": (
        "Agora faz sentido organizar alojamento, transportes, itinerário e orçamento.",
        "Now it makes sense to organise accommodation, transport, itinerary and budget.",
    ),
    "task.travel.where_to_start": ("Por onde queres começar?", "Where would you like to start?"),
    "task.travel.budget_scope_excl_flights": (
        " para alojamento, transportes, alimentação e atividades, sem contar os voos",
        " for accommodation, transport, food and activities, not counting flights",
    ),
    "task.travel.budget_set": ("Ficamos então com até {budget} €{scope}.", "So we're working with up to €{budget}{scope}."),
    "task.travel.next_step_accommodation_type": (
        "O próximo passo mais útil é perceber o tipo de alojamento e a zona onde faz sentido ficar.",
        "The most useful next step is figuring out the type of accommodation and area that makes sense to stay in.",
    ),
    "task.travel.dest_dates_set": ("Temos destino e datas definidos.", "We have the destination and dates set."),
    "task.travel.now_organize_full": (
        "Agora precisamos de organizar alojamento, transportes, itinerário e orçamento.",
        "Now we need to organise accommodation, transport, itinerary and budget.",
    ),
    "task.travel.great_have": ("Ótimo, já temos {known}.", "Great, we already have {known}."),
    "task.travel.until_budget": ("até {budget} €", "up to €{budget}"),
    "task.travel.missing_travel_style": (
        "Falta-me perceber melhor o estilo da viagem.",
        "I still need to understand the travel style better.",
    ),
    "task.travel.looking_for_style": ("Então procuramos {travel_style}.", "So we're looking for {travel_style}."),
    "task.travel.perfect_summary": ("Perfeito. {summary}.", "Perfect. {summary}."),
    "task.travel.can_search_destinations": (
        "Já consigo começar a procurar destinos adequados.",
        "I can start looking for suitable destinations now.",
    ),
    "task.travel.missing_travel_preference": (
        "Acho que ainda me falta perceber como gostas de viajar.",
        "I think I still need to understand how you like to travel.",
    ),
    "task.travel.sure": ("Claro.", "Sure."),
    "task.travel.right_that_helps": ("Certo, isso já ajuda.", "Right, that helps."),
    "task.travel.q.style": (
        "Procuras mais descanso, conhecer sítios novos ou um equilíbrio para descansar e explorar?",
        "Are you after more rest, discovering new places, or a balance of relaxing and exploring?",
    ),
    "task.travel.q.days": ("Quantos dias tens para a viagem?", "How many days do you have for the trip?"),
    "task.travel.q.budget": ("Tens algum limite de orçamento?", "Do you have a budget limit?"),
    "task.travel.q.flight_constraint": (
        "Tens alguma limitação importante em relação à duração dos voos?",
        "Do you have any important constraints on flight duration?",
    ),
    "task.travel.basis_flight_pref": ("preferência por voos curtos", "a preference for short flights"),
    "task.travel.basis_have": ("Temos {parts}", "We have {parts}"),
    "task.travel.basis_initial": ("Temos uma base inicial", "We have an initial basis"),
    "task.known.empty": (
        "Ainda estamos no início. Para já só sei que estamos a trabalhar nesta tarefa.",
        "We're still at the beginning. For now I just know we're working on this task.",
    ),
    "task.known.comparing": ("Estamos a comparar {options}.", "We're comparing {options}."),
    "task.known.travel_mode": ("Até agora, a viagem está em modo de {mode}.", "So far, the trip is in {mode} mode."),
    "task.known.mode_planning": ("planeamento", "planning"),
    "task.known.mode_discovery": ("descoberta", "discovery"),
    "task.known.generic_context": ("Até agora, temos este contexto principal.", "So far, we have this main context."),
    "task.unknowns.none": (
        "Neste momento não vejo nenhuma lacuna essencial. Podemos começar a decidir ou aprofundar um ponto específico.",
        "Right now I don't see any essential gap. We can start deciding or dig deeper into a specific point.",
    ),
    "task.unknowns.missing": ("Ainda falta perceber sobretudo {gaps}.", "We still mainly need to understand {gaps}."),
    "task.decisions.registered": (
        "Já há uma decisão registada: {decisions}.",
        "There's already a decision on record: {decisions}.",
    ),
    "task.decisions.comparison_missing": (
        "Ainda falta decidir a opção final. Antes disso, convém fechar {gaps}.",
        "The final option is still to be decided. Before that, it's worth closing {gaps}.",
    ),
    "task.decisions.comparison_pending": ("Falta decidir qual opção escolhes.", "Which option to choose is still to be decided."),
    "task.decisions.travel_plan": (
        "Ainda falta decidir alojamento, transportes, itinerário e custos.",
        "Accommodation, transport, itinerary and costs are still to be decided.",
    ),
    "task.decisions.travel_destination": (
        "Ainda falta decidir o destino ou o tipo de destino.",
        "The destination or type of destination is still to be decided.",
    ),
    "task.decisions.generic": (
        "Ainda falta transformar isto numa decisão concreta.",
        "This still needs to become a concrete decision.",
    ),
    "task.recommendation.no_confidence": (
        "Ainda não escolheria uma opção com confiança. Falta-nos sobretudo {gaps}.",
        "I wouldn't confidently pick an option yet. We mainly still need {gaps}.",
    ),
    "task.recommendation.more_data": ("mais alguns dados", "a bit more data"),
    "task.recommendation.provisional": (
        "Neste momento, escolheria {recommendation}, de forma provisória, porque {reasons}. "
        "Isto pode mudar se mudares prioridades ou aparecerem dados novos.",
        "Right now, I'd provisionally go with {recommendation}, because {reasons}. "
        "This could change if your priorities shift or new data comes up.",
    ),
    "task.recommendation.default_reason": ("encaixa melhor nos dados atuais", "it fits the current data best"),
    "task.travel_rec.no_confidence": (
        "Ainda não sugeria destinos com confiança. Falta perceber sobretudo {gaps}.",
        "I wouldn't confidently suggest destinations yet. We mainly still need to understand {gaps}.",
    ),
    "task.travel_rec.can_start_planning": (
        "Já dá para começar a planear, mas antes fechava {gaps}.",
        "We can start planning now, but I'd close {gaps} first.",
    ),
    "task.travel_rec.ready": (
        "Já temos base para começar. Eu compararia alojamento, transportes, itinerário e custos antes de fechar o plano.",
        "We have a basis to start. I'd compare accommodation, transport, itinerary and costs before finalising the plan.",
    ),
    "task.why.no_data": (
        "Porque ainda não há dados suficientes para justificar uma escolha com segurança.",
        "Because there isn't enough data yet to safely justify a choice.",
    ),
    "task.why.default_reason": (
        "é a opção com melhor equilíbrio nos dados atuais",
        "it's the option with the best balance given the current data",
    ),
    "task.why.because": (
        "Porque {reasons}. A conclusão continua provisória.",
        "Because {reasons}. The conclusion is still provisional.",
    ),
    "task.next_step.travel_organize": (
        "Agora temos de organizar alojamento, transportes, itinerário e custos.",
        "Now we need to organise accommodation, transport, itinerary and costs.",
    ),
    "task.next_step.close_gaps": ("O próximo passo é fechar {gaps}.", "The next step is to close {gaps}."),
    "task.next_step.choose_direction": (
        "O próximo passo é escolher uma direção e transformar isto em plano.",
        "The next step is to choose a direction and turn this into a plan.",
    ),
    "task.generic.that_helps": ("Certo, isso já ajuda.", "Right, that helps."),
    "task.generic.sure": ("Certo.", "Right."),
    "task.generic.next_step_prompt": (
        "Diz-me o próximo passo concreto quando quiseres.",
        "Tell me the next concrete step whenever you're ready.",
    ),
    "task.generic.continue_parts": ("Certo. Continuamos por partes.", "Right. Let's keep going step by step."),
    "task.comparison.continue_parts": (
        "Certo. Continuamos a comparação por partes.",
        "Right. Let's keep comparing step by step.",
    ),
    "task.card_summary.options": ("Opções: {options}", "Options: {options}"),
    "task.card_summary.limit": ("Limite: {value}", "Limit: {value}"),
    "task.card_summary.priority": ("Prioridade: {value}", "Priority: {value}"),
    "task.card_summary.open": ("Em aberto: {items}", "Open: {items}"),
    # --- Phase 7.7.1: conversation.py _generic_activity_question ----------
    "task.activity_question.plan_trip_with_destination": (
        "Acho que ainda me falta perceber como gostas de viajar. Procuras mais descansar, conhecer sítios novos ou alguma aventura?",
        "I think I still need to understand how you like to travel. Are you looking to rest more, discover new places, or some adventure?",
    ),
    "task.activity_question.plan_trip_no_destination": (
        "Boa ideia. Antes de escolhermos sítios, procuras mais descansar, conhecer sítios novos ou alguma aventura?",
        "Good idea. Before we choose places, are you looking to rest more, discover new places, or some adventure?",
    ),
    "task.activity_question.make_decision": (
        "Certo. Qual é o critério que pesa mais nessa decisão?",
        "Right. Which criterion matters most in this decision?",
    ),
    "task.activity_question.study_topic": (
        "Vamos a isso. Que disciplina é? Já tens apontamentos ou vais começar pelos materiais da cadeira?",
        "Let's get to it. What subject is it? Do you already have notes, or will you start from the course materials?",
    ),
    "task.activity_question.research_topic": (
        "Certo. Queres uma visão geral primeiro ou queres comparar opções concretas?",
        "Right. Do you want a general overview first, or do you want to compare specific options?",
    ),
    "task.activity_question.generic_comparison": (
        "Certo. Já temos essa comparação aberta.",
        "Right. We already have that comparison open.",
    ),
    "task.activity_question.generic_activity": (
        "Certo. Vamos trabalhando nisso por partes.",
        "Right. Let's work through it step by step.",
    ),
    "task.activity_question.writing": ("Claro. O que queres escrever?", "Sure. What would you like to write?"),
    "task.activity_question.default": (
        "Certo. Diz-me só qual é o próximo passo.",
        "Right. Just tell me what the next step is.",
    ),
    # --- Phase 7.7.1: _available_information_summary() fixed headers ------
    "info_summary.intro": (
        "Tenho acesso a estas fontes de informação:",
        "I have access to these sources of information:",
    ),
    "info_summary.permanent_memory_header": ("\nMemória permanente:", "\nPermanent memory:"),
    "info_summary.no_permanent_memory": (
        "- Ainda não encontrei memória permanente relevante.",
        "- I haven't found any relevant permanent memory yet.",
    ),
    "info_summary.tasks_header": ("\nTarefas:", "\nTasks:"),
    "info_summary.timeline_header": ("\nTimeline:", "\nTimeline:"),
    "info_summary.workspace_files_header": ("\nFicheiros da workspace:", "\nWorkspace files:"),
    "info_summary.workspace_tool_missing": (
        "- A ferramenta de listagem da workspace não está ligada.",
        "- The workspace listing tool isn't connected.",
    ),
    "info_summary.observed_context_header": (
        "\nContexto observado do computador:",
        "\nObserved computer context:",
    ),
    "info_summary.no_observed_context": (
        "- Ainda não tenho contexto observado disponível.",
        "- I don't have any observed context available yet.",
    ),
    # --- Phase 7.7.1: _deterministic_help_response / travel followup ------
    # found via a REAL scenario run (section 3's exact example), not the
    # original grep-based audit -- a separate deterministic fast-path that
    # fires BEFORE the task system engages for phrases like "preciso de
    # planear as minhas ferias".
    "help.pyside6_missing": (
        "Esse erro quer dizer que o PySide6 não está instalado no ambiente Python que estás a usar. "
        "Instala-o no venv com: pip install PySide6",
        "That error means PySide6 isn't installed in the Python environment you're using. "
        "Install it in the venv with: pip install PySide6",
    ),
    "help.travel_planning_intro": (
        "Boa ideia. Antes de sugerir sítios, deixa-me perceber uma coisa: "
        "procuras mais descansar, conhecer sítios novos ou alguma aventura?",
        "Good idea. Before suggesting places, let me understand one thing: "
        "are you looking to rest more, discover new places, or some adventure?",
    ),
    "help.travel_destination_followup": (
        "Acho que ainda me falta perceber como gostas de viajar. "
        "Isso vai influenciar mais a escolha do que saber apenas a zona.",
        "I think I still need to understand how you like to travel. "
        "That will shape the choice more than just knowing the area.",
    ),
    # --- Phase 7.7.2: generic_comparison dialogue subsystem ----------------
    # (_plan_comparison_dialogue and every helper it calls). Canonical
    # values (option names, budget numbers, predicates) are never
    # translated -- only the Echo-authored sentence scaffolding is.
    "task.comparison.active": (
        "Certo. Temos a comparação entre {options} ativa.",
        "Right. We have the comparison between {options} active.",
    ),
    "task.comparison.which_options": ("Certo. Que opções queres comparar?", "Right. Which options do you want to compare?"),
    "task.comparison.budget_and_battery_change": (
        "Isso muda a comparação. O limite passa para {budget} € e a bateria passa a pesar menos.",
        "That changes the comparison. The limit becomes €{budget} and the battery matters less.",
    ),
    "task.comparison.budget_set": (
        "Perfeito. Fica definido um limite de {budget} €.",
        "Perfect. A limit of €{budget} is set.",
    ),
    "task.comparison.default_ack": (
        "Certo, isso fica como contexto da comparação.",
        "Right, that becomes context for the comparison.",
    ),
    "task.comparison.q.criteria": (
        "Que fator pesa mais para ti nesta escolha?",
        "Which factor matters most to you in this choice?",
    ),
    "task.comparison.q.know_value": ("Sabes o {value}?", "Do you know the {value}?"),
    "task.comparison.q.missing_reference": (
        "Falta perceber a {value}. Tens alguma referência?",
        "Still need to understand the {value}. Do you have any reference?",
    ),
    "task.comparison.q.missing_decisive": (
        "Falta perceber a {value}. Isso é decisivo para ti?",
        "Still need to understand the {value}. Is that a deciding factor for you?",
    ),
    "task.comparison.q.missing_compare": ("Falta comparar o {value}.", "Still need to compare the {value}."),
    "task.comparison.q.missing_generic": ("Falta perceber {value}.", "Still need to understand {value}."),
    "task.comparison.budget_inside_plural": ("{options} ficam dentro do limite", "{options} are within the limit"),
    "task.comparison.budget_inside_singular": ("{option} fica dentro do limite", "{option} is within the limit"),
    "task.comparison.budget_over": ("{option} fica {amount} € acima", "{option} is €{amount} over"),
    "task.comparison.preference_camera_and_battery": (
        "Percebi: a câmara pesa mais, e a bateria fica como fator secundário.",
        "Got it: the camera matters more, and the battery becomes a secondary factor.",
    ),
    "task.comparison.preference_camera": (
        "Percebi: a câmara passa a pesar mais na decisão.",
        "Got it: the camera now matters more in the decision.",
    ),
    "task.comparison.preference_battery_low": (
        "Percebi: a bateria passa a pesar menos na decisão.",
        "Got it: the battery now matters less in the decision.",
    ),
    "task.comparison.preference_battery_other": (
        "Percebi: a bateria também conta na decisão.",
        "Got it: the battery matters too in the decision.",
    ),
    "task.comparison.preference_battery_first": (
        "Percebi: a bateria conta na decisão.",
        "Got it: the battery matters in the decision.",
    ),
    "task.comparison.preference_support": (
        "Percebi: suporte passa a pesar mais do que escolher apenas pelo preço.",
        "Got it: support now matters more than choosing on price alone.",
    ),
    "task.comparison.preference_generic": ("Percebi a prioridade.", "Got it, noted the priority."),
    "task.comparison.gaining_ground": (
        "Com os dados atuais, {recommendation} começa a ganhar força.",
        "With the current data, {recommendation} is starting to gain ground.",
    ),
    "task.comparison.need_compare_priority": (
        "Agora precisamos de comparar essa prioridade nas opções.",
        "Now we need to compare that priority across the options.",
    ),
    "task.comparison.more_price_data": ("Certo, já temos mais dados de preço.", "Right, we now have more price data."),
    "task.comparison.prices_known_plural": ("Já temos os preços: {list}.", "We now have the prices: {list}."),
    "task.comparison.prices_known_singular": ("Já temos o preço: {list}.", "We now have the price: {list}."),
    "task.comparison.tradeoff_clearer": (
        "Boa, já temos um tradeoff mais claro: {items}.",
        "Good, we now have a clearer trade-off: {items}.",
    ),
    "task.comparison.tradeoff_helps": ("Certo, isso ajuda a comparar melhor.", "Right, that helps compare better."),
    "task.comparison.recommendation_gaining": (
        "Neste momento, {recommendation} parece mais forte, porque {reasons}.",
        "Right now, {recommendation} looks stronger, because {reasons}.",
    ),
    "task.comparison.still_provisional": ("Ainda é provisório: falta {gaps}.", "Still provisional: we still need {gaps}."),
    "task.quality.battery_advantage": ("Vantagem de autonomia", "Battery life advantage"),
    "task.quality.battery_value": ("Autonomia: {value}", "Battery life: {value}"),
    "task.quality.camera_better": ("Parece ter vantagem na câmara", "Seems to have a camera advantage"),
    "task.quality.camera_good": ("Boa prestação de câmara", "Good camera performance"),
    "task.quality.camera_mixed": ("Evidência mista sobre a câmara", "Mixed evidence on the camera"),
    "task.quality.camera_little_evidence": (
        "Ainda há pouca evidência sobre a câmara",
        "There's still little evidence on the camera",
    ),
    "task.quality.support_better": ("Melhor suporte", "Better support"),
    "task.quality.support_limited": ("Suporte limitado", "Limited support"),
    "task.quality.label.camera": ("Câmara", "Camera"),
    "task.quality.label.battery": ("Autonomia", "Battery"),
    "task.quality.label.support": ("Suporte", "Support"),
    "task.quality.summary.weak_positive": (
        "{label}: sinais positivos, ainda com evidência limitada.",
        "{label}: positive signs, though evidence is still limited.",
    ),
    "task.quality.summary.advantage": ("{label}: parece haver vantagem.", "{label}: there seems to be an advantage."),
    "task.quality.summary.good_mixed": (
        "{label}: resultados positivos, mas com alguma incerteza entre fontes.",
        "{label}: positive results, but with some uncertainty between sources.",
    ),
    "task.quality.summary.good": ("{label}: boa prestação.", "{label}: good performance."),
    "task.quality.summary.mixed": ("{label}: evidência mista.", "{label}: mixed evidence."),
    "task.quality.reason.strong": (
        "tem vantagem de {label} em fonte técnica forte",
        "has a {label} advantage in a strong technical source",
    ),
    "task.quality.reason.useful": (
        "tem sinais úteis de vantagem em {label}",
        "has useful signs of a {label} advantage",
    ),
    "task.quality.reason.weak": ("tem sinais fracos de vantagem em {label}", "has weak signs of a {label} advantage"),
    "task.quality.reason.some": ("tem sinais de vantagem em {label}", "has signs of a {label} advantage"),
    # --- Phase 7.7.2: evidence / sources / conflicts formatters ------------
    # Source titles, quotations, and any source-derived text are NEVER
    # translated (section 6) -- only Echo-authored wrappers/labels/
    # connective sentences are.
    "task.important.empty": (
        "O essencial é simples: ainda não temos dados suficientes para decidir bem.",
        "It's simple: we still don't have enough data to decide well.",
    ),
    "task.important.budget_limit": ("o limite é {budget} €", "the limit is €{budget}"),
    "task.important.recommendation_now": (
        "{recommendation} parece melhor neste momento",
        "{recommendation} looks best right now",
    ),
    "task.important.biggest_gap": ("a maior lacuna é {gap}", "the biggest gap is {gap}"),
    "task.important.comparison_header": ("O mais importante: {parts}.", "The most important thing: {parts}."),
    "task.important.travel_missing": ("falta {gap}", "missing {gap}"),
    "task.important.header": ("O essencial: {parts}.", "The essentials: {parts}."),
    "task.sources.none": ("Ainda não tenho uma fonte concreta para isso.", "I don't have a concrete source for that yet."),
    "task.sources.from_web": ("Veio de {title}: “{evidence}”", "It came from {title}: “{evidence}”"),
    "task.sources.web_reference_suffix": (" ({reference}).", " ({reference})."),
    "task.sources.default_web_label": ("fonte web", "web source"),
    "task.sources.from_user": ("Veio do que me disseste: “{text}”.", "It came from what you told me: “{text}”."),
    "task.evidence.no_sources": (
        "Ainda não tenho fontes suficientes para responder a isso com segurança.",
        "I don't have enough sources yet to answer that safely.",
    ),
    "task.evidence.comparative_no_proof": (
        "Não. Estas fontes ajudam a avaliar {known}, mas ainda não provam que seja melhor do que {missing}.",
        "No. These sources help assess {known}, but don't yet prove it's better than {missing}.",
    ),
    "task.evidence.one_of_the_options": ("uma das opções", "one of the options"),
    "task.evidence.no_direct_evidence": (
        "Ainda não tenho evidência direta suficiente sobre {target}.",
        "I don't have enough direct evidence about {target} yet.",
    ),
    "task.evidence.that_point": ("esse ponto", "that point"),
    "task.evidence.sufficiency_missing": (
        "Chega para ter uma primeira leitura sobre {target}, mas ainda não para fechar a comparação. "
        "Falta evidência equivalente sobre {missing}.",
        "That's enough for a first read on {target}, but not enough to close the comparison. "
        "We're still missing equivalent evidence about {missing}.",
    ),
    "task.evidence.sufficiency_yes": (
        "Sim, já chega para uma conclusão provisória. Eu ainda a trataria como provisória, "
        "porque podemos reforçar com mais uma fonte independente.",
        "Yes, that's enough for a provisional conclusion. I'd still treat it as provisional, "
        "since we could strengthen it with one more independent source.",
    ),
    "task.evidence.sufficiency_no": (
        "Ainda não. As fontes que temos são demasiado fracas para decidir com segurança.",
        "Not yet. The sources we have are too weak to decide safely.",
    ),
    "task.evidence.strong_positive": (
        "Sim. Com estas fontes, já há boa evidência de que {target} tem {criterion}.",
        "Yes. With these sources, there's already good evidence that {target} has {criterion}.",
    ),
    "task.evidence.strong_mixed": (
        "As fontes sugerem que {target} tem pontos fortes em {criterion}, mas a conclusão ainda é mista.",
        "The sources suggest {target} has strengths in {criterion}, but the conclusion is still mixed.",
    ),
    "task.evidence.strong_useful_base": (
        "Estas fontes já dão uma base útil sobre {target}, mas eu manteria a conclusão como provisória.",
        "These sources already give a useful basis on {target}, but I'd keep the conclusion provisional.",
    ),
    "task.evidence.weak_signals": (
        "Há sinais sobre {target}, mas a evidência ainda é fraca. Eu não fecharia uma decisão só com isto.",
        "There are signs about {target}, but the evidence is still weak. I wouldn't close a decision on this alone.",
    ),
    "task.evidence.not_enough": (
        "Ainda não tenho evidência suficiente para concluir isso com segurança.",
        "I don't have enough evidence yet to safely conclude that.",
    ),
    "task.predicate_label.camera_quality": ("boa câmara", "a good camera"),
    "task.predicate_label.battery": ("boa autonomia", "good battery life"),
    "task.predicate_label.price": ("um preço definido", "a set price"),
    "task.predicate_label.support_quality": ("bom suporte", "good support"),
    "task.conflicts.none": (
        "Não vejo informação contraditória ativa neste momento.",
        "I don't see any active contradictory information right now.",
    ),
    "task.conflicts.found": (
        "Há valores diferentes nestes pontos: {items}.",
        "There are differing values on these points: {items}.",
    ),
    "task.reason.within_budget": (
        "fica dentro do orçamento ({price} <= {budget})",
        "is within budget ({price} <= {budget})",
    ),
    "task.reason.over_budget": ("passa o orçamento ({price} > {budget})", "goes over budget ({price} > {budget})"),
    "task.reason.limited_support": ("tem suporte mais limitado", "has more limited support"),
    "task.reason.options_to_compare": ("opções a comparar", "options to compare"),
    # --- Phase 7.7.2: infer_unknowns() -- feeds EVERY dialogue/query
    # formatter above (travel, comparison, generic); found only by running
    # a real mid-comparison scenario, not the original audit. option/{}
    # values are canonical, never translated -- only the descriptive noun
    # phrase around them localizes.
    "task.unknown.price_of": ("preço de {option}", "price of {option}"),
    "task.unknown.camera_of": ("câmara de {option}", "camera of {option}"),
    "task.unknown.battery_of": ("autonomia de {option}", "battery life of {option}"),
    "task.unknown.decision_criteria": ("critérios de decisão", "decision criteria"),
    "task.unknown.destination_type": ("tipo de destino", "type of destination"),
    "task.unknown.dates_or_duration": ("datas ou duração", "dates or duration"),
    "task.unknown.budget": ("orçamento", "budget"),
    "task.unknown.travel_pace": ("ritmo da viagem", "trip pace"),
    "task.unknown.accommodation": ("alojamento", "accommodation"),
    "task.unknown.transport": ("transportes", "transport"),
    "task.unknown.itinerary": ("itinerário", "itinerary"),
    "task.unknown.costs": ("custos", "costs"),
    "task.unknown.next_concrete_step": ("próximo passo concreto", "next concrete step"),
    "task.predicate_label.price_bare": ("preço", "price"),
}


def localize(key: str, response_language: str, **params: object) -> str:
    """The single localization boundary for Echo-authored presentation
    strings (section 17) -- callers pass a stable key (never the raw
    Portuguese text) plus any interpolation params; this picks pt-PT or
    en-GB from `_STRINGS` and formats it. Raises KeyError for an unknown
    key rather than silently falling back, so a missing translation is a
    loud bug, not a quiet Portuguese leak."""
    template = _STRINGS[key][1 if normalize_response_language(response_language) == ENGLISH_GB else 0]
    return template.format(**params) if params else template


_BUILT_IN_TASK_GOALS = {
    "escrever um email": "task.goal.compose_email",
    "escrever em conjunto": "task.goal.writing",
    "planear ferias": "task.goal.plan_trip",
    "tomar uma decisao": "task.goal.make_decision",
    "estudar": "task.goal.study_topic",
    "investigar opcoes": "task.goal.research_topic",
    "comparar opcoes": "task.goal.generic_comparison",
    "trabalhar num objetivo": "task.goal.generic_activity",
    "continuar a compreender preferencias antes de recomendar": "task.goal.preference_discovery",
}


def localize_task_goal(goal: str, language: str) -> str:
    """Display text for a task goal: built-in goals (stored as fixed
    Portuguese values) are localized; a goal the user stated is shown as is.
    The stored value itself never changes."""
    key = _BUILT_IN_TASK_GOALS.get(_normalize(str(goal or "")).strip())
    return localize(key, language) if key else str(goal or "")


def localize_deterministic_response(response: str, language: str) -> str:
    if normalize_response_language(language) != ENGLISH_GB:
        return response
    text = str(response or "").strip()
    normalized = _normalize(text)
    if not text:
        return text
    exact = {
        "claro. o que se passa?": "Sure. What's going on?",
        "feito.": "Done.",
        "de nada.": "You're welcome.",
        # Phase 7.3.4: compose_email task-dialogue questions (conversation.py
        # _compose_email_question) -- a technical/deterministic turn, so it
        # never reaches the LLM/language_instruction path at all; without an
        # entry here it always leaked out in Portuguese regardless of
        # response_language.
        "claro. para quem e?": "Sure. Who's it for?",
        "o que precisas de pedir ou dizer nesse email?": "What do you need to ask for or say in that email?",
        "sobre que assunto, mais especificamente?": "What subject, more specifically?",
    }
    if normalized in exact:
        return exact[normalized]
    if normalized.startswith("boa, temos ") and " como opcoes" in normalized:
        body = re.sub(r"^boa, temos\s+", "", text, flags=re.IGNORECASE)
        body = re.sub(r"\s+como opções.*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+como opcoes.*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+e\s+([^,]+)$", r" and \1", body)
        return f"Good, we have {body} as options. They're already in the workspace."
    if normalized.startswith("certo. temos a comparacao entre "):
        body = re.sub(r"^Certo\.\s*Temos a comparação entre\s+", "", text)
        body = re.sub(r"\s+ativa\.$", "", body)
        return f"Right. The comparison between {body} is active."
    if normalized.startswith("ainda falta perceber"):
        return "I still need to understand one or two things before making a useful suggestion."
    if normalized.startswith("ainda nao tenho fontes suficientes"):
        return "I don't have enough reliable sources to answer that safely yet."
    if (
        normalized.startswith("nao tenho ainda uma ferramenta de pesquisa")
        or normalized.startswith("ainda nao tenho uma ferramenta de pesquisa")
        or normalized.startswith("nao tenho uma ferramenta de pesquisa")
    ):
        return "I don't have a real research tool connected yet."
    return text


def _asks_for_english(text: str) -> bool:
    if text in {"in english", "english", "english please"}:
        return True
    return any(
        phrase in text
        for phrase in (
            "prefiro ingles",
            "fala em ingles",
            "fala comigo em ingles",
            "responde em ingles",
            "resposta em ingles",
            "quero que fales ingles",
            "a partir de agora fala ingles",
            "muda o idioma para ingles",
            "speak english",
            "answer in english",
            "switch to english",
            "respond in english",
        )
    )


def _asks_for_portuguese(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "prefiro portugues",
            "fala portugues",
            "fala em portugues",
            "responde em portugues",
            "resposta em portugues",
            "volta para portugues",
            "volta ao portugues",
            "muda o idioma para portugues",
            "switch to portuguese",
            "speak portuguese",
            "answer in portuguese",
            "respond in portuguese",
        )
    )


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(without_marks.split())
