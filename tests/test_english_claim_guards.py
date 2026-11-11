"""English ungrounded-claim guards (cherry-picked fddf667) matched the exact
phrases from one conversation ("i was just waiting for you", "just the
usual system checks"), so other wordings of Echo describing activity of its
own passed through; and "open" as a request marker also matched questions
and comments ("What window is open?", "open-source tools"), so a reply
there could be treated as a false workspace-operation claim.
"""
from __future__ import annotations

import pytest

from assistant.conversation import (
    _detect_ungrounded_activity_claim,
    _detect_ungrounded_memory_claim,
    _detect_ungrounded_workspace_operation_claim,
)

INVENTIONS = (
    "I've been running some checks",
    "I was waiting for you to come back",
    "I spent the morning organising files",
    "I've been busy reviewing your emails",
    "I just finished updating the system",
    "I was thinking about our last chat",
    # Real reply from the 11/11 log (no first person).
    "Not much to report, just the usual system checks.",
    "I’ve been running some checks",  # curly apostrophe, as the model writes it
)
LEGITIMATE = (
    "What window is open?",
    "I can open the file if you want",
    "You said you were working from home",
    "open-source tools are great",
)


@pytest.mark.parametrize("response", INVENTIONS)
def test_echo_describing_its_own_activity_is_blocked(response: str) -> None:
    assert _detect_ungrounded_activity_claim(response), response


@pytest.mark.parametrize("text", LEGITIMATE)
def test_legitimate_statements_pass_the_activity_and_memory_guards(text: str) -> None:
    assert not _detect_ungrounded_activity_claim(text)
    assert not _detect_ungrounded_memory_claim(text)


@pytest.mark.parametrize("message", LEGITIMATE)
def test_open_in_a_question_or_comment_is_not_an_open_request(message: str) -> None:
    # Only a request to open makes "I opened ..." a false operation claim.
    assert not _detect_ungrounded_workspace_operation_claim(message, "I opened it.")


@pytest.mark.parametrize("message", ("Open the email", "Can you open the report?", "Please open it"))
def test_open_requests_still_catch_false_success_claims(message: str) -> None:
    assert _detect_ungrounded_workspace_operation_claim(message, "I opened it.")
