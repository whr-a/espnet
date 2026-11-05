"""Speaker, verification, long-form, multi-speaker and mixed rewards and contracts."""

import json

import pytest

from espnet2.rl.batch import collate_rollouts
from espnet2.rl.tasks.manifest import read_manifest
from espnet2.rl.rewards import REWARDS, get_reward, outcome_rewards
from espnet2.rl.rewards.asr import longform_wer_reward, repeated_word_fraction
from espnet2.rl.rewards.asr import word_error_reward
from espnet2.rl.rewards.cpwer import (
    MAX_CPWER_SPEAKERS,
    concatenated_wer,
    multispeaker_cpwer_reward,
    split_speaker_streams,
)
from espnet2.rl.rewards.mixed import mixed_task_reward
from espnet2.rl.rewards.speaker import (
    match_speaker,
    speaker_id_reward,
    speaker_verification_reward,
)

CHOICES = ["1263", "374", "8842", "2277"]


def speaker_source(**updates):
    return {
        "id": "spk-1",
        "mode": "audio_text",
        "speaker": "1263",
        "speaker_choices": list(CHOICES),
        **updates,
    }


def longform_source(words=250, **updates):
    return {
        "id": "lf-1",
        "mode": "audio_text",
        "reference": " ".join(f"w{index}" for index in range(words)),
        "min_reference_words": 200,
        **updates,
    }


def verification_source(verdict="same", identity="a__b__4s"):
    return {
        "id": identity,
        "verdict": verdict,
        "verdict_choices": ["same", "different"],
        "reward_task": "speaker_verification",
    }


def verification_score(text, verdict="same"):
    return speaker_verification_reward(verification_source(verdict), {"text": text})


def multispeaker_source(**updates):
    return {
        "id": "ms-1",
        "mode": "audio_text",
        "references": ["hello there my friend", "i am leaving now"],
        "speaker_tags": ["<spk0>", "<spk1>"],
        **updates,
    }


# --------------------------------------------------------------- speaker id


def test_speaker_label_is_matched_as_words_not_as_a_substring():
    # "1263" must not be found inside a longer number, and a stated label is
    # recognized without an answer tag.
    assert match_speaker("the speaker is 1263", CHOICES) == (0, "unique")
    assert match_speaker("speaker 12634", CHOICES) == (None, "none")
    assert match_speaker("either 374 or 8842", CHOICES) == (None, "ambiguous")


def test_speaker_reward_scores_accuracy_plus_a_terminal_format_bonus():
    source = speaker_source()
    exact = speaker_id_reward(source, {"text": "Low voice. <answer>1263</answer>"})
    assert exact["reward"] == pytest.approx(1.1)
    assert (exact["acc"], exact["format"], exact["path_tag"]) == (1.0, 1.0, 1.0)

    wrong = speaker_id_reward(source, {"text": "<answer>374</answer>"})
    assert wrong["reward"] == pytest.approx(0.1)
    assert (wrong["acc"], wrong["parsed"]) == (0.0, 1.0)

    untagged = speaker_id_reward(source, {"text": "I think it is 1263"})
    assert untagged["reward"] == pytest.approx(1.0)
    assert (untagged["format"], untagged["path_line"]) == (0.0, 1.0)

    # Naming two candidates earns no accuracy: the response chose nobody.
    hedged = speaker_id_reward(source, {"text": "<answer>374 or 8842</answer>"})
    assert (hedged["acc"], hedged["ambiguous"], hedged["reward"]) == (
        0.0,
        1.0,
        pytest.approx(0.1),
    )

    silent = speaker_id_reward(source, {"text": ""})
    assert (silent["acc"], silent["parsed"], silent["reward"]) == (0.0, 0.0, 0.0)


def test_speaker_reward_rejects_a_label_outside_its_own_choice_set():
    with pytest.raises(ValueError, match="not exactly one of its speaker_choices"):
        speaker_id_reward(speaker_source(speaker="9999"), {"text": "<answer>374</answer>"})


def test_speaker_reward_rejects_a_rollout_that_did_not_emit_text():
    with pytest.raises(ValueError, match="emits other actions"):
        speaker_id_reward(speaker_source(mode="text_audio"), {"text": "1263"})


def test_speaker_manifest_without_supervision_is_rejected_before_training(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"id": "spk-1", "reference": "the cat"}\n')
    with pytest.raises(ValueError, match="speaker_id requires speaker"):
        read_manifest(
            path,
            config=type(
                "C", (), {"reward_function": "speaker_id"}
            )(),
        )


# ------------------------------------------------------- speaker verification


def test_verification_tagged_correct_answer_earns_accuracy_and_format():
    scores = verification_score("The voices match. <answer>same</answer>", "same")
    assert scores["acc"] == 1.0
    assert scores["format"] == 1.0
    assert scores["parsed"] == 1.0
    assert scores["reward"] == pytest.approx(1.1)
    assert scores["path_tag"] == 1.0


def test_verification_tagged_wrong_answer_keeps_only_the_format_bonus():
    scores = verification_score("<answer>different</answer>", "same")
    assert scores["acc"] == 0.0
    assert scores["format"] == 1.0
    assert scores["reward"] == pytest.approx(0.1)


def test_verification_naming_both_labels_is_ambiguous_and_earns_nothing():
    scores = verification_score(
        "<answer>not the same, they are different</answer>", "different"
    )
    assert scores["ambiguous"] == 1.0
    assert scores["parsed"] == 0.0
    assert scores["acc"] == 0.0


def test_verification_falls_back_to_the_last_line_without_a_tag():
    scores = verification_score(
        "A long caption about the recording.\nThe speakers are different.",
        "different",
    )
    assert scores["path_line"] == 1.0
    assert scores["format"] == 0.0
    assert scores["acc"] == 1.0
    assert scores["reward"] == pytest.approx(1.0)


def test_verification_two_answer_blocks_lose_the_format_bonus():
    scores = verification_score(
        "<answer>same</answer> or maybe <answer>same</answer>", "same"
    )
    assert scores["format"] == 0.0
    assert scores["acc"] == 1.0


def test_per_class_columns_reconstruct_tpr_tnr_and_balanced_accuracy():
    # A policy that always answers "same": accuracy 0.5 on a balanced set, but
    # balanced accuracy exactly 0.5 -- which is the point of these columns.
    batch = [
        verification_score("<answer>same</answer>", verdict)
        for verdict in ["same"] * 4 + ["different"] * 4
    ]

    def mean(name):
        return sum(row[name] for row in batch) / len(batch)

    assert mean("acc") == pytest.approx(0.5)
    tpr = mean("hit_same") / mean("true_same")
    tnr = mean("hit_different") / mean("true_different")
    assert tpr == pytest.approx(1.0)
    assert tnr == pytest.approx(0.0)
    assert (tpr + tnr) / 2 == pytest.approx(0.5)
    assert mean("said_same") == pytest.approx(1.0)


def test_a_discriminating_policy_moves_balanced_accuracy_off_one_half():
    batch = [
        verification_score("<answer>same</answer>", "same"),
        verification_score("<answer>same</answer>", "same"),
        verification_score("<answer>different</answer>", "different"),
        verification_score("<answer>same</answer>", "different"),
    ]

    def mean(name):
        return sum(row[name] for row in batch) / len(batch)

    tpr = mean("hit_same") / mean("true_same")
    tnr = mean("hit_different") / mean("true_different")
    assert (tpr + tnr) / 2 == pytest.approx(0.75)


def test_verification_contract_rejects_a_manifest_without_verdicts():
    with pytest.raises(ValueError, match="verdict"):
        speaker_verification_reward.validate_sources([{"id": "x"}])


def test_verification_contract_rejects_a_foreign_reward_task():
    rows = [{**verification_source(), "reward_task": "speaker_id"}]
    with pytest.raises(ValueError, match="declares reward_task"):
        speaker_verification_reward.validate_sources(rows)


def test_verification_contract_rejects_choices_that_are_not_the_two_labels():
    rows = [{**verification_source(), "verdict_choices": ["yes", "no"]}]
    with pytest.raises(ValueError, match="not"):
        speaker_verification_reward.validate_sources(rows)


def test_verification_contract_accepts_the_real_manifest_shape():
    speaker_verification_reward.validate_sources(
        [verification_source("same", "a__b__4s"),
         verification_source("different", "c__d__4s")]
    )


def test_a_verdict_outside_its_own_choices_is_rejected_at_scoring_time():
    bad = {**verification_source(), "verdict": "maybe"}
    with pytest.raises(ValueError, match="not exactly one of"):
        speaker_verification_reward(bad, {"text": "<answer>same</answer>"})


# ---------------------------------------------------------------- long-form


def test_longform_reward_separates_early_stopping_from_looping():
    source = longform_source(words=200)
    half = " ".join(f"w{index}" for index in range(100))
    stopped = longform_wer_reward(source, {"text": half, "finish_reason": "stop"})
    # 100 deletions over 200 reference words.
    assert stopped["wer"] == pytest.approx(0.5)
    assert stopped["reward"] == pytest.approx(-0.5)
    assert stopped["length_ratio"] == pytest.approx(0.5)
    assert (stopped["truncated"], stopped["repeat_rate"]) == (0.0, 0.0)

    looped = longform_wer_reward(
        source, {"text": " ".join(["w0 w1 w2 w3 w4 w5 w6 w7 w8"] * 30),
                 "finish_reason": {"type": "length"}}
    )
    assert looped["truncated"] == 1.0
    assert looped["repeat_rate"] > 0.9
    assert looped["length_ratio"] > 1.0
    # However many words a loop inserts, it never scores below silence.
    assert looped["reward"] == pytest.approx(-1.0)

    empty = longform_wer_reward(source, {"text": ""})
    assert (empty["empty"], empty["reward"]) == (1.0, pytest.approx(-1.0))


def test_repeat_rate_is_zero_for_a_response_shorter_than_its_window():
    assert repeated_word_fraction("one two three") == 0.0


def test_longform_contract_refuses_to_relabel_short_utterances(tmp_path):
    config = type(
        "C", (), {"reward_function": "longform_wer"}
    )()
    short = tmp_path / "short.jsonl"
    short.write_text(
        '{"id": "lf-1", "reference": "the cat sat", "min_reference_words": 200}\n'
    )
    with pytest.raises(ValueError, match="below the declared long-form floor"):
        read_manifest(short, config=config)

    undeclared = tmp_path / "undeclared.jsonl"
    undeclared.write_text('{"id": "lf-1", "reference": "the cat sat"}\n')
    with pytest.raises(ValueError, match="requires reference, min_reference_words"):
        read_manifest(undeclared, config=config)

    good = tmp_path / "good.jsonl"
    good.write_text(
        '{"id": "lf-1", "reference": "%s", "min_reference_words": 200}\n'
        % " ".join(f"w{index}" for index in range(200))
    )
    assert len(read_manifest(good, config=config)) == 1


def test_longform_contract_rejects_audio_past_its_declared_window(tmp_path):
    """A reference-length floor alone does not make an item trainable.

    OWSM's load_audio raises on audio longer than its window rather than
    truncating, so a manifest packed by utterance count can die on every
    sample at the first read. This catches it on CPU instead.
    """
    config = type(
        "C", (), {"reward_function": "longform_wer"}
    )()
    reference = " ".join(f"w{index}" for index in range(200))
    path = tmp_path / "over.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "lf-1",
                "reference": reference,
                "min_reference_words": 200,
                "audio_seconds": 114.15,
                "max_audio_seconds": 30.0,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="exceed the audio window"):
        read_manifest(path, config=config)

    within = tmp_path / "within.jsonl"
    within.write_text(
        json.dumps(
            {
                "id": "lf-1",
                "reference": reference,
                "min_reference_words": 200,
                "audio_seconds": 25.1,
                "max_audio_seconds": 30.0,
            }
        )
        + "\n"
    )
    assert len(read_manifest(within, config=config)) == 1

    # A manifest from before the window was recorded still reads, so an
    # in-flight run is not broken by this check; it only fails at first read.
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(
            {"id": "lf-1", "reference": reference, "min_reference_words": 200}
        )
        + "\n"
    )
    assert len(read_manifest(legacy, config=config)) == 1


# ------------------------------------------------------- multi-speaker cpWER


def test_cpwer_is_invariant_to_which_tag_a_speaker_was_given():
    source = multispeaker_source()
    first, second = source["references"]
    ordered = multispeaker_cpwer_reward(
        source, {"text": f"<spk0> {first} <spk1> {second}"}
    )
    swapped = multispeaker_cpwer_reward(
        source, {"text": f"<spk0> {second} <spk1> {first}"}
    )
    assert ordered["cpwer"] == swapped["cpwer"] == 0.0
    assert ordered["reward"] == swapped["reward"] == 0.0
    assert ordered["tag_format"] == 1.0


def test_cpwer_counts_a_missing_speaker_as_whole_stream_deletions():
    source = multispeaker_source()
    first, _ = source["references"]
    row = multispeaker_cpwer_reward(source, {"text": f"<spk0> {first}"})
    # Four words of speaker one are right; all four of speaker two are missing.
    assert row["cpwer"] == pytest.approx(4 / 8)
    assert (row["tag_format"], row["empty_streams"]) == (0.0, 1.0)


def test_cpwer_does_not_credit_a_preamble_to_the_first_speaker():
    source = multispeaker_source()
    first, second = source["references"]
    row = multispeaker_cpwer_reward(
        source, {"text": f"Sure, here it is. <spk0> {first} <spk1> {second}"}
    )
    assert row["cpwer"] == 0.0
    assert row["unassigned_words"] == 4.0


def test_cpwer_joins_repeated_turns_of_one_speaker_in_order():
    streams, preamble, stated = split_speaker_streams(
        "<spk0> hello <spk1> hi <spk0> again", ["<spk0>", "<spk1>"]
    )
    assert streams == {"<spk0>": "hello again", "<spk1>": "hi"}
    assert (preamble, stated) == ("", ["<spk0>", "<spk1>", "<spk0>"])


def test_cpwer_normalizes_by_total_reference_words_not_per_speaker():
    # One long and one short reference: a short speaker's errors cannot
    # dominate, which per-speaker averaging would allow.
    assert concatenated_wer(["a b c d e f g h", "i"], ["a b c d e f g h", "i"]) == 0.0
    assert concatenated_wer(["a b c d e f g h", "i"], ["a b c d e f g h", "x"]) == (
        pytest.approx(1 / 9)
    )


@pytest.mark.parametrize(
    "record, message",
    [
        ({"references": ["only one"], "speaker_tags": ["<spk0>"]}, "at least two"),
        (
            {"references": ["a", "b"], "speaker_tags": ["<spk0>", "<spk0>"]},
            "must be distinct",
        ),
        (
            {"references": ["a", "b"], "speaker_tags": ["<spk0>", "speaker1"]},
            "must be distinct",
        ),
        (
            {"references": ["a", "   "], "speaker_tags": ["<spk0>", "<spk1>"]},
            "empty speaker reference",
        ),
    ],
)
def test_cpwer_contract_rejects_an_unscoreable_manifest(record, message):
    with pytest.raises(ValueError, match=message):
        multispeaker_cpwer_reward.validate_sources([{"id": "ms-1", **record}])


def test_cpwer_refuses_more_speakers_than_its_search_can_afford():
    """cpWER's cost is factorial in the speaker count, once per candidate."""
    count = MAX_CPWER_SPEAKERS + 1
    record = {
        "id": "ms-many",
        "references": [f"speaker {index}" for index in range(count)],
        "speaker_tags": [f"<spk{index}>" for index in range(count)],
    }
    with pytest.raises(ValueError, match=f"accepts at most {MAX_CPWER_SPEAKERS}"):
        multispeaker_cpwer_reward.validate_sources([record])
    # The largest accepted mixture still validates and scores.
    ok = {
        "id": "ms-max",
        "mode": "audio_text",
        "references": [f"speaker {index}" for index in range(MAX_CPWER_SPEAKERS)],
        "speaker_tags": [f"<spk{index}>" for index in range(MAX_CPWER_SPEAKERS)],
    }
    multispeaker_cpwer_reward.validate_sources([ok])
    text = " ".join(
        f"<spk{index}> speaker {index}" for index in range(MAX_CPWER_SPEAKERS)
    )
    assert multispeaker_cpwer_reward(ok, {"text": text})["cpwer"] == 0.0


# ----------------------------------------------------------- mixed-task mix


def batched_component(sources, generations):
    batched_component.calls.append(len(sources))
    return [{"reward": -float(len(generation["text"]))} for generation in generations]


batched_component.batched = True
batched_component.calls = []


def test_every_record_is_scored_by_the_task_it_declares():
    sources = [
        speaker_source(id="a", reward_task="speaker_id"),
        longform_source(id="b", reward_task="longform_wer", words=200),
        speaker_source(id="c", reward_task="speaker_id"),
    ]
    generations = [
        {"text": "<answer>1263</answer>"},
        {"text": " ".join(f"w{index}" for index in range(200))},
        {"text": "<answer>374</answer>"},
    ]
    rows = mixed_task_reward(sources, generations)
    assert [row["reward"] for row in rows] == [
        pytest.approx(1.1),
        pytest.approx(0.0),
        pytest.approx(0.1),
    ]
    # One fixed column set, so outcome_rewards can log batch means.
    assert {tuple(sorted(row)) for row in rows} == {
        (
            "reward",
            "share_longform_wer",
            "share_speaker_id",
            "sum_longform_wer",
            "sum_speaker_id",
        )
    }
    share = sum(row["share_speaker_id"] for row in rows) / len(rows)
    total = sum(row["sum_speaker_id"] for row in rows) / len(rows)
    assert total / share == pytest.approx(0.6)  # (1.1 + 0.1) / 2


def test_a_mixed_manifest_may_name_a_reward_outside_the_builtin_tasks():
    task = f"{__name__}:batched_component"
    assert get_reward(task) is batched_component
    batched_component.calls.clear()
    rows = mixed_task_reward(
        [speaker_source(id="a", reward_task=task), speaker_source(id="b", reward_task=task)],
        [{"text": "abc"}, {"text": "de"}],
    )
    # A batched component still sees its whole subset in one call.
    assert batched_component.calls == [2]
    assert [row["reward"] for row in rows] == [-3.0, -2.0]


def test_an_unknown_reward_name_lists_the_registered_names():
    with pytest.raises(ValueError, match="Unknown reward 'speaker-identification'"):
        get_reward("speaker-identification")
    # The task names the preparers stamp into reward_task stay registered.
    assert {"asr_wer", "longform_wer", "speaker_id", "multispeaker_cpwer"} <= set(
        REWARDS
    )


def test_every_registered_reward_resolves_and_no_name_is_the_default_wer():
    for name in REWARDS:
        assert callable(get_reward(name)), name
    assert get_reward(None) is get_reward("wer") is word_error_reward
    assert get_reward("espnet2.rl.rewards.asr:asr_wer_reward") is get_reward(
        "asr_wer"
    )


def test_mixed_contract_delegates_to_each_component_before_training(tmp_path):
    config = type(
        "C", (), {"reward_function": "mixed"}
    )()
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"id": "a", "reward_task": "speaker_id", "speaker": "1263",'
        ' "speaker_choices": ["1263", "374"]}\n'
        '{"id": "b", "reward_task": "longform_wer", "reference": "the cat sat",'
        ' "min_reference_words": 200}\n'
    )
    with pytest.raises(ValueError, match="below the declared long-form floor"):
        read_manifest(path, config=config)

    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"id": "a", "speaker": "1263"}\n')
    with pytest.raises(ValueError, match="needs reward_task in every record"):
        read_manifest(missing, config=config)


def test_mixed_rewards_reach_the_trainer_through_outcome_rewards():
    sources = [
        speaker_source(id="a", reward_task="speaker_id"),
        longform_source(id="b", reward_task="longform_wer", words=200),
    ]
    texts = [
        "<answer>1263</answer>",
        " ".join(f"w{index}" for index in range(100)),
    ]
    generations = [
        {
            "source_id": source["id"],
            "token_ids": [7, 7],
            "log_probs": [-1.0, -1.0],
            "prompt_token_ids": [1],
            "text": text,
        }
        for source, text in zip(sources, texts, strict=True)
    ]
    batch = collate_rollouts(sources, generations, 1)
    rewards, token_rewards, metrics = outcome_rewards(
        batch, "mixed"
    )
    assert rewards.tolist() == [pytest.approx(1.1), pytest.approx(-0.5)]
    # The reward lands on each candidate's last sampled action.
    assert token_rewards[0].tolist() == [0.0, pytest.approx(1.1)]
    assert metrics["reward/share_speaker_id"] == pytest.approx(0.5)
    assert metrics["reward/sum_longform_wer"] == pytest.approx(-0.25)


def test_speaker_contract_rejects_a_candidate_list_that_leaks_the_label(tmp_path):
    """The gap this closes: no reward can see how the data was built.

    A per-speaker-constant candidate list makes the list a function of the
    label whenever training and heldout share one speaker inventory, which is
    the only closed-set split a disjoint-speaker corpus allows.
    """
    config = type(
        "C", (), {"reward_function": "speaker_id"}
    )()

    def manifest(name, records):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    leaky = [
        {
            "id": f"{speaker}-c-{index:04d}",
            "speaker": speaker,
            # same list, same order, for every utterance of this speaker
            "speaker_choices": ["A", "B"] if speaker == "A" else ["B", "A"],
        }
        for speaker in ("A", "B")
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="identifies the answer"):
        read_manifest(manifest("leaky.jsonl", leaky), config=config)

    # Declaring the per-speaker draw is an acknowledged choice, not an accident.
    acknowledged = [{**r, "choice_key": "speaker"} for r in leaky]
    assert len(read_manifest(manifest("ack.jsonl", acknowledged), config=config)) == 6

    # Varying the list per utterance closes the channel.
    safe = [
        {**r, "speaker_choices": (["A", "B"] if i % 2 else ["B", "A"])}
        for i, r in enumerate(leaky)
    ]
    assert len(read_manifest(manifest("safe.jsonl", safe), config=config)) == 6

    # One utterance per speaker cannot leak anything, so it must still pass.
    single = [
        {"id": "A-c-0000", "speaker": "A", "speaker_choices": ["A", "B"]},
        {"id": "B-c-0000", "speaker": "B", "speaker_choices": ["B", "A"]},
    ]
    assert len(read_manifest(manifest("single.jsonl", single), config=config)) == 2


def test_a_single_speaker_manifest_is_not_a_leak(tmp_path):
    """With one label there is nothing for the candidate list to distinguish."""
    config = type(
        "C", (), {"reward_function": "speaker_id"}
    )()
    path = tmp_path / "one.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"374-c-{index:04d}",
                    "speaker": "374",
                    "speaker_choices": ["374", "2277"],
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    assert len(read_manifest(path, config=config)) == 3


def test_a_preparer_that_declares_its_draw_as_choice_draw_is_understood(tmp_path):
    """The declaration has two spellings, and both must be read.

    ``choice_key`` is what espnet2.rl.tasks.speaker stamps. A preparer written
    before that field existed records the same thing as
    ``choice_draw: {"keyed_on": ...}`` instead, and the speaker-id run on
    LibriSpeech dev-clean produced its manifests that way. Reading only
    ``choice_key`` would let a deliberate per-speaker draw be rejected for
    not declaring itself, while it had.
    """
    config = type(
        "C", (), {"reward_function": "speaker_id"}
    )()

    def manifest(name, records):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    constant = [
        {
            "id": f"{speaker}-c-{index:04d}",
            "speaker": speaker,
            "speaker_choices": ["A", "B"] if speaker == "A" else ["B", "A"],
        }
        for speaker in ("A", "B")
        for index in range(3)
    ]

    # Spelled the old way, saying "speaker": a deliberate draw, let through.
    read_manifest(
        manifest(
            "declared.jsonl",
            [{**r, "choice_draw": {"keyed_on": "speaker"}} for r in constant],
        ),
        config=config,
    )

    # Spelled the old way, saying "utterance_id": the declaration claims the
    # safe draw, so a per-speaker-constant list contradicts it and is the
    # accidental case the check exists for.
    with pytest.raises(ValueError, match="identifies the answer"):
        read_manifest(
            manifest(
                "contradicts.jsonl",
                [
                    {**r, "choice_draw": {"keyed_on": "utterance_id", "choices": 2}}
                    for r in constant
                ],
            ),
            config=config,
        )

    # A per-utterance draw spelled the old way: lists vary within a speaker,
    # so nothing is leaked and it passes however it is declared.
    varied = [
        {
            "id": f"{speaker}-v-{index:04d}",
            "speaker": speaker,
            "speaker_choices": [speaker, f"other-{index}"],
            "choice_draw": {"keyed_on": "utterance_id", "seed": 1, "choices": 2},
        }
        for speaker in ("A", "B")
        for index in range(3)
    ]
    read_manifest(manifest("varied.jsonl", varied), config=config)


def test_a_choice_draw_that_is_not_a_mapping_does_not_crash_the_check(tmp_path):
    """The field is read off records this code did not write, so its shape is
    not guaranteed. A string there must not turn a data check into a
    TypeError at manifest-read time."""
    config = type(
        "C", (), {"reward_function": "speaker_id"}
    )()
    path = tmp_path / "weird.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"{speaker}-w-{index:04d}",
                    "speaker": speaker,
                    "speaker_choices": [speaker, f"other-{index}"],
                    "choice_draw": "utterance_id",
                }
            )
            + "\n"
            for speaker in ("A", "B")
            for index in range(3)
        )
    )
    read_manifest(path, config=config)
