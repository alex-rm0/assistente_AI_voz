from assistant.cognition.preference_builder import UserPreferenceBuilder


def test_travel_planning_prefers_question_before_recommendation() -> None:
    assessment = UserPreferenceBuilder().assess("Queria planear umas férias.")

    assert assessment.domain == "travel"
    assert not assessment.enough_for_recommendation
    assert "descansar" in assessment.next_question


def test_road_trip_builds_preference_model() -> None:
    assessment = UserPreferenceBuilder().assess("Gostava de fazer uma road trip.")

    assert "quer fazer uma road trip" in assessment.known_preferences
    assert "prefere experiências a destinos isolados" in assessment.inferred_preferences
    assert "dormir sempre no mesmo sítio" in assessment.next_question


def test_travelling_with_partner_adds_useful_inference() -> None:
    assessment = UserPreferenceBuilder().assess("Vou com a minha namorada.", recent_context="Queria planear umas férias.")

    assert assessment.domain == "travel"
    assert "vai viajar acompanhado" in assessment.known_preferences
    assert "a viagem deve funcionar bem para duas pessoas" in assessment.inferred_preferences


def test_laptop_choice_asks_for_use_before_recommending() -> None:
    assessment = UserPreferenceBuilder().assess("Preciso de escolher um portátil.")

    assert assessment.domain == "laptop"
    assert not assessment.enough_for_recommendation
    assert "uso principal" in assessment.missing_preferences


def test_home_choice_asks_for_priority_before_recommending() -> None:
    assessment = UserPreferenceBuilder().assess("Queria procurar uma casa.")

    assert assessment.domain == "home"
    assert not assessment.enough_for_recommendation
    assert "localização" in assessment.next_question


def test_car_choice_asks_for_use_before_recommending() -> None:
    assessment = UserPreferenceBuilder().assess("Ajuda-me a escolher um carro.")

    assert assessment.domain == "car"
    assert not assessment.enough_for_recommendation
    assert "uso principal" in assessment.next_question


def test_study_planning_asks_before_plan() -> None:
    assessment = UserPreferenceBuilder().assess("Tenho de estudar para um exame.")

    assert assessment.domain == "study"
    assert not assessment.enough_for_recommendation
    assert "Quando é o exame?" in assessment.next_question
