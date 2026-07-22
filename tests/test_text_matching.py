from __future__ import annotations

from assistant.text_matching import contains_any_phrase, contains_phrase, find_evidence_span, find_near_pair_span, normalize_for_matching


def test_contains_phrase_does_not_match_substring_inside_another_word() -> None:
    text = normalize_for_matching("não vou fazer braço de ferro")
    assert not contains_phrase(text, "erro")


def test_contains_phrase_matches_the_real_word() -> None:
    text = normalize_for_matching("ocorreu um erro")
    assert contains_phrase(text, "erro")


def test_contains_phrase_matches_multi_word_phrase_regardless_of_accents() -> None:
    text = normalize_for_matching("Atividade recente do computador")
    assert contains_phrase(text, "atividade recente")


def test_contains_any_phrase() -> None:
    text = normalize_for_matching("janela ativa agora")
    assert contains_any_phrase(text, ("programa ativo", "janela ativa"))
    assert not contains_any_phrase(text, ("programa ativo", "app ativa"))


def test_find_evidence_span_returns_first_match_or_empty() -> None:
    text = normalize_for_matching("Quero saber o que estive a fazer ontem")
    assert find_evidence_span(text, ("o que estive a fazer", "atividade recente")) == "o que estive a fazer"
    assert find_evidence_span(text, ("atividade recente",)) == ""


def test_find_near_pair_span_tolerates_a_verb_between_subject_and_state() -> None:
    text = normalize_for_matching("Que janelas tens detetadas?")
    span = find_near_pair_span(text, ("janela", "janelas"), ("abertas", "detetadas"))
    assert span


def test_find_near_pair_span_respects_word_boundaries() -> None:
    # "aplicacoes" must not match a state word that happens to share letters
    # with an unrelated nearby word.
    text = normalize_for_matching("aplicacoes para o meu curso")
    span = find_near_pair_span(text, ("aplicacao", "aplicacoes"), ("abertas", "detetadas"))
    assert span == ""
