"""The WER every family's reward and heldout evaluation share."""

import pytest

from espnet2.rl.rewards import word_error_rate, word_error_reward, words


def test_punctuation_and_case_are_not_word_errors():
    # A real OWSM v4 dev-clean hypothesis against its LibriSpeech reference:
    # every former "error" was punctuation or the hyphen in to-morrow.
    reference = (
        "HE WOULD SEE HOW THINGS TURNED OUT TO MORROW AND THEN HE WOULD TALK TO"
        " HER THEY WERE GOING TO MEET AS USUAL"
    )
    hypothesis = (
        "He would see how things turned out to-morrow, and then he would talk to"
        " her. They were going to meet as usual."
    )
    assert word_error_rate(reference, hypothesis) == 0
    assert word_error_reward({"reference": reference}, {"text": hypothesis}) == 0


def test_real_word_errors_still_count():
    assert word_error_rate("THE CAT SAT", "the cat sat down") == pytest.approx(1 / 3)
    assert word_error_rate("THE CAT SAT", "a cat") == pytest.approx(2 / 3)
    assert word_error_rate("WIFE'S ACTION", "wife's action.") == 0
    # An empty reference counts every hypothesis word against one word.
    assert word_error_rate("", "anything else") == 2


def test_words_keep_letters_and_digits_of_any_script():
    assert words("Café, naïve 42!") == ["café", "naïve", "42"]
    assert words("你好，世界") == ["你好", "世界"]
