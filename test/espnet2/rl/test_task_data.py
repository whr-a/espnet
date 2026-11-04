"""Manifest preparation for the speaker, long-form and multi-speaker tasks."""

import json
import sys

import numpy as np
import pytest
import soundfile as sf

from espnet2.rl import tasks
from espnet2.rl.rewards.asr import longform_wer_reward
from espnet2.rl.rewards.cpwer import multispeaker_cpwer_reward
from espnet2.rl.rewards.mixed import mixed_task_reward
from espnet2.rl.rewards.speaker import speaker_id_reward
from espnet2.rl.tasks.derive import append_text_instruction, derive_task_manifest
from espnet2.rl.tasks.longform import (
    audio_window_seconds,
    prepare_longform_manifest,
    session_of,
)
from espnet2.rl.tasks.multispeaker import (
    MULTISPEAKER_PROMPT,
    MULTISPEAKER_PROMPTS,
    MULTISPEAKER_VERBATIM_PROMPT,
    mix_two_speakers,
    prepare_multispeaker_manifest,
)
from espnet2.rl.tasks.speaker import prepare_speaker_id_manifest, speaker_choice_set

SPEAKERS = ("spkA", "spkB", "spkC", "spkD")
SENTENCE = "hello world this is a test of long form speech"


def corpus(tmp_path, per_speaker=2, seconds=1.0, sentences=1):
    """A tiny Kaldi corpus: wav.scp, text and utt2spk over four speakers."""
    audio = tmp_path / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    waves, texts, speakers = [], [], []
    for speaker in SPEAKERS:
        for index in range(per_speaker):
            identity = f"{speaker}-sess1-{index:04d}"
            path = audio / f"{identity}.wav"
            sf.write(
                str(path),
                generator.normal(0, 0.05, int(16000 * seconds)).astype("float32"),
                16000,
                subtype="FLOAT",
            )
            waves.append(f"{identity} {path}")
            texts.append(f"{identity} " + " ".join([SENTENCE] * sentences))
            speakers.append(f"{identity} {speaker}")
    for name, rows in (("wav.scp", waves), ("text", texts), ("utt2spk", speakers)):
        (tmp_path / name).write_text("\n".join(rows) + "\n")
    return tmp_path / "wav.scp", tmp_path / "text", tmp_path / "utt2spk"


# ---------------------------------------------------------------- speaker id


def test_speaker_choices_are_seeded_and_always_contain_the_true_speaker():
    first = speaker_choice_set("spkA", SPEAKERS, 4, 42)
    assert sorted(first) == sorted(SPEAKERS)
    assert speaker_choice_set("spkA", SPEAKERS, 4, 42) == first
    # A different seed may reorder the same inventory; the set never changes.
    assert sorted(speaker_choice_set("spkA", SPEAKERS, 4, 7)) == sorted(first)
    # The answer is not pinned to one position across speakers.
    positions = {
        speaker: speaker_choice_set(speaker, SPEAKERS, 4, 42).index(speaker)
        for speaker in SPEAKERS
    }
    assert len(set(positions.values())) > 1
    with pytest.raises(ValueError, match="needs 4 distinct speakers"):
        speaker_choice_set("spkA", ("spkA", "spkB"), 4, 42)


def test_speaker_choices_keyed_per_speaker_identify_the_answer_on_their_own():
    """The default draw repeats one list per speaker, so the list leaks the label.

    Harmless when train and heldout hold different speakers; fatal for a split
    that keeps one inventory and divides it by utterance, which is why
    choice_key exists.
    """
    lists = {
        speaker: speaker_choice_set(speaker, SPEAKERS, 3, 42) for speaker in SPEAKERS
    }
    # One ordered list per speaker, and the map from list to label is injective.
    assert len({tuple(value) for value in lists.values()}) == len(SPEAKERS)
    # Two utterances of one speaker would be offered exactly the same list.
    assert speaker_choice_set("spkA", SPEAKERS, 3, 42, None) == lists["spkA"]


def test_speaker_choices_keyed_per_utterance_stop_leaking_the_answer():
    first = speaker_choice_set("spkA", SPEAKERS, 3, 42, "spkA-utt-1")
    second = speaker_choice_set("spkA", SPEAKERS, 3, 42, "spkA-utt-2")
    for options in (first, second):
        assert "spkA" in options and len(options) == 3
    # Same speaker, different utterance: a different list, so the list alone no
    # longer names the answer.
    assert first != second
    # Still deterministic for a given utterance.
    assert speaker_choice_set("spkA", SPEAKERS, 3, 42, "spkA-utt-1") == first


def test_speaker_manifest_can_key_its_choices_per_utterance(tmp_path):
    """Contrast the two keyings on one corpus: only "speaker" repeats a list."""
    wav, _, utt2spk = corpus(tmp_path, per_speaker=5)

    def lists_per_speaker(name, **kwargs):
        records = prepare_speaker_id_manifest(
            wav, utt2spk, tmp_path / name, choices=3, seed=42, **kwargs
        )
        speaker_id_reward.validate_sources(records)
        seen = {}
        for record in records:
            assert record["speaker"] in record["speaker_choices"]
            for choice in record["speaker_choices"]:
                assert choice in record["prompt"]
            seen.setdefault(record["speaker"], set()).add(
                tuple(record["speaker_choices"])
            )
        return seen

    per_speaker = lists_per_speaker("by_speaker.jsonl", choice_key="speaker")
    per_utterance = lists_per_speaker("by_utt.jsonl", choice_key="utterance")
    # Keyed on the speaker, all five utterances share one ordered list, so the
    # list names the answer by itself.
    assert all(len(value) == 1 for value in per_speaker.values())
    # Keyed on the utterance the lists vary within a speaker. A tiny inventory
    # lets two draws coincide, so require variation rather than five distinct.
    assert any(len(value) > 1 for value in per_utterance.values())
    assert sum(len(value) for value in per_utterance.values()) > sum(
        len(value) for value in per_speaker.values()
    )
    audit = json.loads((tmp_path / "by_utt.jsonl.audit.json").read_text())
    assert audit["choice_key"] == "utterance"
    assert json.loads((tmp_path / "by_speaker.jsonl.audit.json").read_text())[
        "choice_key"
    ] == "speaker"


def test_speaker_manifest_rejects_an_unknown_choice_key(tmp_path):
    wav, _, utt2spk = corpus(tmp_path)
    with pytest.raises(ValueError, match="choice_key must be speaker or utterance"):
        prepare_speaker_id_manifest(
            wav, utt2spk, tmp_path / "spk.jsonl", choice_key="chapter"
        )


def test_speaker_manifest_carries_its_own_prompt_and_passes_the_reward_contract(tmp_path):
    wav, _, utt2spk = corpus(tmp_path)
    records = prepare_speaker_id_manifest(
        wav, utt2spk, tmp_path / "spk.jsonl", choices=3, seed=42
    )
    assert len(records) == len(SPEAKERS) * 2
    speaker_id_reward.validate_sources(records)
    for record in records:
        assert record["reward_task"] == "speaker_id"
        assert record["speaker"] in record["speaker_choices"]
        assert len(record["speaker_choices"]) == 3
        # The prompt names exactly the choices the reward will score against.
        for choice in record["speaker_choices"]:
            assert choice in record["prompt"]
        # The speaker label is supervision, not an ASR reference.
        assert "reference" not in record
    audit = json.loads((tmp_path / "spk.jsonl.audit.json").read_text())
    assert audit["task"] == "speaker_id"
    assert audit["speakers"] == len(SPEAKERS)
    assert audit["rows"] == len(records)
    assert set(audit) >= {"wav_scp_sha256", "utt2spk_sha256", "manifest_sha256"}


def test_a_prepared_manifest_is_never_overwritten(tmp_path):
    wav, _, utt2spk = corpus(tmp_path)
    prepare_speaker_id_manifest(wav, utt2spk, tmp_path / "spk.jsonl")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_speaker_id_manifest(wav, utt2spk, tmp_path / "spk.jsonl")


def test_speaker_preparation_requires_matching_utterance_ids(tmp_path):
    wav, _, utt2spk = corpus(tmp_path)
    utt2spk.write_text(utt2spk.read_text().split("\n", 1)[1])
    with pytest.raises(ValueError, match="same utterance IDs"):
        prepare_speaker_id_manifest(wav, utt2spk, tmp_path / "spk.jsonl")


# ---------------------------------------------------------------- long-form


def test_session_id_strips_only_the_utterance_index():
    assert session_of("374-180298-0000") == "374-180298"


def test_concatenation_builds_long_references_and_keeps_its_sources(tmp_path):
    wav, text, _ = corpus(tmp_path, per_speaker=4, sentences=12)
    records = prepare_longform_manifest(
        wav,
        text,
        tmp_path / "lf.jsonl",
        min_reference_words=200,
        concatenate=2,
        max_audio_seconds=30.0,
    )
    # Four utterances per speaker, taken two at a time, over four speakers.
    assert len(records) == 8
    longform_wer_reward.validate_sources(records)
    for record in records:
        assert record["reward_task"] == "longform_wer"
        assert len(record["reference"].split()) >= 200
        assert len(record["concatenated_ids"]) == 2
        waveform, rate = sf.read(record["audio"], dtype="float32")
        assert rate == 16000
        # Two one-second sources concatenate to two seconds of audio.
        assert len(waveform) == pytest.approx(32000, abs=2)
    audit = json.loads((tmp_path / "lf.jsonl.audit.json").read_text())
    assert audit["concatenate"] == 2
    assert min(audit["reference_words"].values()) >= 200


def test_longform_preparation_refuses_a_short_corpus(tmp_path):
    wav, text, _ = corpus(tmp_path, sentences=1)
    with pytest.raises(ValueError, match="shorter than the declared long-form floor"):
        prepare_longform_manifest(
            wav,
            text,
            tmp_path / "lf.jsonl",
            min_reference_words=200,
            max_audio_seconds=30.0,
        )


def test_longform_preparation_reports_when_no_session_is_long_enough(tmp_path):
    wav, text, _ = corpus(tmp_path, per_speaker=2, sentences=12)
    with pytest.raises(ValueError, match="consecutive utterances to concatenate"):
        prepare_longform_manifest(
            wav,
            text,
            tmp_path / "lf.jsonl",
            min_reference_words=10,
            concatenate=3,
            max_audio_seconds=30.0,
        )


# ------------------------------------------------- the model's audio window


def test_the_window_is_read_from_the_models_own_training_yaml(tmp_path):
    recipe = tmp_path / "train.yaml"
    recipe.write_text("preprocessor_conf:\n  speech_length: 30\n  fs: 16000\n")
    assert audio_window_seconds(recipe) == 30.0
    # A frontend may carry it instead.
    other = tmp_path / "frontend.yaml"
    other.write_text("frontend_conf:\n  speech_length: 20.5\n")
    assert audio_window_seconds(other) == 20.5
    # A family that declares no window says so, rather than implying none.
    silent = tmp_path / "silent.yaml"
    silent.write_text("corelm_conf:\n  hf_model_tag: x\n")
    assert audio_window_seconds(silent) is None


def test_longform_preparation_refuses_to_guess_the_window(tmp_path):
    """The hole this closes: a preparer that does not know the model's window
    can build a set whose every sample dies at the first read of training."""
    wav, text, _ = corpus(tmp_path, per_speaker=2, sentences=12)
    with pytest.raises(ValueError, match="needs the model's audio window"):
        prepare_longform_manifest(
            wav, text, tmp_path / "lf.jsonl", min_reference_words=10
        )


def test_longform_preparation_rejects_audio_past_the_window(tmp_path):
    # One second per utterance, concatenated three at a time, against a
    # two-second window: every packed item is a second too long.
    wav, text, _ = corpus(tmp_path, per_speaker=3, seconds=1.0, sentences=12)
    with pytest.raises(ValueError, match="exceed the declared 2.0 s audio window"):
        prepare_longform_manifest(
            wav,
            text,
            tmp_path / "lf.jsonl",
            min_reference_words=10,
            concatenate=3,
            max_audio_seconds=2.0,
        )


def test_longform_records_carry_the_window_they_were_built_against(tmp_path):
    wav, text, _ = corpus(tmp_path, per_speaker=4, seconds=1.0, sentences=12)
    records = prepare_longform_manifest(
        wav,
        text,
        tmp_path / "lf.jsonl",
        min_reference_words=10,
        concatenate=2,
        max_audio_seconds=30.0,
    )
    for record in records:
        assert record["max_audio_seconds"] == 30.0
        assert record["audio_seconds"] == pytest.approx(2.0, abs=0.01)
    longform_wer_reward.validate_sources(records)
    audit = json.loads((tmp_path / "lf.jsonl.audit.json").read_text())
    assert audit["max_audio_seconds"] == 30.0


def test_a_training_yaml_supplies_the_window_without_repeating_it(tmp_path):
    recipe = tmp_path / "train.yaml"
    recipe.write_text("preprocessor_conf:\n  speech_length: 30\n")
    wav, text, _ = corpus(tmp_path, per_speaker=2, seconds=1.0, sentences=12)
    records = prepare_longform_manifest(
        wav, text, tmp_path / "lf.jsonl", min_reference_words=10, train_config=recipe
    )
    assert all(record["max_audio_seconds"] == 30.0 for record in records)


# ------------------------------------------------------- multi-speaker mixing


def test_mixing_delays_the_second_speaker_and_never_clips():
    first = np.full(16000, 0.8, dtype="float32")
    second = np.full(16000, 0.8, dtype="float32")
    mixture, onset = mix_two_speakers(first, second, 0.5, 16000)
    assert onset == pytest.approx(0.5)
    assert len(mixture) == 24000
    assert np.abs(mixture).max() <= 1.0
    # Fully overlapped audio stays the same length as its longest input.
    together, onset = mix_two_speakers(first, second, 1.0, 16000)
    assert (len(together), onset) == (16000, 0.0)
    # Overlap 0 concatenates.
    apart, _ = mix_two_speakers(first, second, 0.0, 16000)
    assert len(apart) == 32000
    with pytest.raises(ValueError, match=r"overlap must lie in \[0, 1\]"):
        mix_two_speakers(first, second, 1.5, 16000)


def test_mixtures_always_pair_two_different_speakers(tmp_path):
    wav, text, utt2spk = corpus(tmp_path, per_speaker=3)
    records = prepare_multispeaker_manifest(
        wav, text, utt2spk, tmp_path / "ms.jsonl", overlap=0.3, seed=42
    )
    assert len(records) == 6  # 12 utterances, two per mixture
    multispeaker_cpwer_reward.validate_sources(records)
    for record in records:
        assert record["reward_task"] == "multispeaker_cpwer"
        assert record["speakers"][0] != record["speakers"][1]
        assert len(record["references"]) == len(record["speaker_tags"]) == 2
        assert record["second_onset_seconds"] == pytest.approx(0.7, abs=1e-3)
        for tag in record["speaker_tags"]:
            assert tag in record["prompt"]
    audit = json.loads((tmp_path / "ms.jsonl.audit.json").read_text())
    assert (audit["task"], audit["overlap"], audit["speakers"]) == (
        "multispeaker_cpwer",
        0.3,
        4,
    )


def test_mixing_needs_more_than_one_speaker(tmp_path):
    wav, text, utt2spk = corpus(tmp_path)
    utt2spk.write_text(
        "\n".join(
            f"{line.split()[0]} only" for line in utt2spk.read_text().splitlines()
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="needs utterances from two speakers"):
        prepare_multispeaker_manifest(wav, text, utt2spk, tmp_path / "ms.jsonl")


def test_the_default_instruction_is_what_a_mixture_carries_unless_asked(tmp_path):
    wav, text, utt2spk = corpus(tmp_path, per_speaker=3)
    records = prepare_multispeaker_manifest(wav, text, utt2spk, tmp_path / "d.jsonl")
    audit = json.loads((tmp_path / "d.jsonl.audit.json").read_text())
    assert audit["prompt_template"] == MULTISPEAKER_PROMPT
    for record in records:
        assert record["prompt"] == MULTISPEAKER_PROMPT.format(
            tags=" and ".join(record["speaker_tags"])
        )


def test_the_verbatim_instruction_names_each_tag_on_its_own(tmp_path):
    """verbatim_tagged needs per-tag fields the default template never used.

    The default instruction only interpolates the tags joined with "and", so a
    template that addresses one speaker at a time could not be rendered at all
    before. Both spellings must keep working from the same preparer, because
    the choice between them decides whether cpWER has any spread to learn
    from on a captioner-pretrained policy.
    """
    wav, text, utt2spk = corpus(tmp_path, per_speaker=3)
    records = prepare_multispeaker_manifest(
        wav,
        text,
        utt2spk,
        tmp_path / "vt.jsonl",
        prompt=MULTISPEAKER_PROMPTS["verbatim_tagged"],
    )
    assert MULTISPEAKER_PROMPTS["verbatim_tagged"] is MULTISPEAKER_VERBATIM_PROMPT
    audit = json.loads((tmp_path / "vt.jsonl.audit.json").read_text())
    assert audit["prompt_template"] == MULTISPEAKER_VERBATIM_PROMPT
    for record in records:
        first, second = record["speaker_tags"]
        # Each tag is placed individually, and no field is left unrendered.
        assert f"Use {first} for one speaker and {second} for the other" in (
            record["prompt"]
        )
        assert "{" not in record["prompt"]
    # The mixtures themselves do not depend on the instruction.
    plain = prepare_multispeaker_manifest(
        wav, text, utt2spk, tmp_path / "plain.jsonl"
    )
    assert [r["audio_sha256"] for r in plain] == [
        r["audio_sha256"] for r in records
    ]


def test_the_cli_offers_exactly_the_named_instructions(tmp_path, monkeypatch):
    wav, text, utt2spk = corpus(tmp_path, per_speaker=3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tasks",
            "multispeaker_cpwer",
            "--wav-scp",
            str(wav),
            "--text",
            str(text),
            "--utt2spk",
            str(utt2spk),
            "--prompt-template",
            "verbatim_tagged",
            "--output",
            str(tmp_path / "cli.jsonl"),
        ],
    )
    tasks.main()
    audit = json.loads((tmp_path / "cli.jsonl.audit.json").read_text())
    assert audit["prompt_template"] == MULTISPEAKER_VERBATIM_PROMPT
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tasks",
            "multispeaker_cpwer",
            "--wav-scp",
            str(wav),
            "--text",
            str(text),
            "--utt2spk",
            str(utt2spk),
            "--prompt-template",
            "no_such_template",
            "--output",
            str(tmp_path / "never.jsonl"),
        ],
    )
    with pytest.raises(SystemExit):
        tasks.main()


# ------------------------------------------------------- derived manifests


def prepared_manifest(tmp_path, rows=2, nq=9):
    path = tmp_path / "prepared.jsonl"
    records = [
        {
            "id": f"374-180298-{index:04d}",
            "reference": "the cat sat",
            "prompt_token_ids": [[5] + [0] * (nq - 1), [64] + [7] * (nq - 1)],
            "prompt_codec_mask": [False, True],
            "native_preparation": {"family": "opuslm", "native_audio_frames": 2},
            "mode": "audio_text",
        }
        for index in range(rows)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path, records


def test_an_instruction_occupies_only_the_primary_stream():
    prompt = [[5, 0, 0], [64, 7, 7]]
    extended = append_text_instruction(prompt, [11, 12], 1000)
    assert extended == [[5, 0, 0], [64, 7, 7], [1011, 0, 0], [1012, 0, 0]]
    # The original matrix is not mutated in place.
    assert prompt == [[5, 0, 0], [64, 7, 7]]
    with pytest.raises(ValueError, match="rectangular"):
        append_text_instruction([[5, 0, 0], [64, 7]], [11], 1000)
    with pytest.raises(ValueError, match="rectangular"):
        append_text_instruction([], [11], 1000)


def test_deriving_a_task_reuses_the_audited_audio_condition(tmp_path):
    path, original = prepared_manifest(tmp_path)
    supervision = {
        record["id"]: {"speaker": "374", "speaker_choices": ["374", "2277"]}
        for record in original
    }
    records = derive_task_manifest(
        [path],
        tmp_path / "derived.jsonl",
        task="speaker_id",
        supervision=supervision,
        instruction_tokens=[11, 12, 13],
        text_token_start=1000,
        drop=("reference",),
    )
    speaker_id_reward.validate_sources(records)
    for record, source in zip(records, original, strict=True):
        assert record["reward_task"] == "speaker_id"
        assert "reference" not in record
        # The condition frames are byte-identical; only the instruction is new.
        assert record["prompt_token_ids"][:2] == source["prompt_token_ids"]
        assert record["prompt_token_ids"][2:] == [
            [1011, 0, 0, 0, 0, 0, 0, 0, 0],
            [1012, 0, 0, 0, 0, 0, 0, 0, 0],
            [1013, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        assert record["prompt_codec_mask"] == [False, True, False, False, False]
        assert record["native_preparation"] == source["native_preparation"]
        assert record["derived_prompt"]["original_frames"] == 2
        assert record["derived_prompt"]["frames"] == 5
    audit = json.loads((tmp_path / "derived.jsonl.audit.json").read_text())
    assert audit["derived_from_sha256"]
    assert audit["instruction_tokens"] == [11, 12, 13]
    assert audit["audio_condition"] == "copied unchanged from the source manifest"


def test_deriving_without_supervision_or_offset_is_rejected(tmp_path):
    path, original = prepared_manifest(tmp_path)
    with pytest.raises(ValueError, match="No speaker_id supervision"):
        derive_task_manifest(
            [path], tmp_path / "a.jsonl", task="speaker_id", supervision={}
        )
    with pytest.raises(ValueError, match="text_token_start"):
        derive_task_manifest(
            [path],
            tmp_path / "b.jsonl",
            task="speaker_id",
            supervision={record["id"]: {} for record in original},
            instruction_tokens=[1],
        )


def test_deriving_rejects_the_same_record_twice(tmp_path):
    path, original = prepared_manifest(tmp_path)
    with pytest.raises(ValueError, match="Duplicate source record"):
        derive_task_manifest(
            [path, path],
            tmp_path / "c.jsonl",
            task="speaker_id",
            supervision={record["id"]: {} for record in original},
        )


def test_supervision_may_name_each_records_own_task(tmp_path):
    """A mixed manifest declares a task per record, not one for the file."""
    path, original = prepared_manifest(tmp_path, rows=2)
    supervision = {
        original[0]["id"]: {"reward_task": "asr_wer"},
        original[1]["id"]: {
            "reward_task": "speaker_id",
            "speaker": "374",
            "speaker_choices": ["374", "2277"],
        },
    }
    records = derive_task_manifest(
        [path], tmp_path / "mixed.jsonl", task="mixed", supervision=supervision
    )
    assert [record["reward_task"] for record in records] == ["asr_wer", "speaker_id"]
    mixed_task_reward.validate_sources(records)


# ------------------------------------------------------------------- the CLI


@pytest.mark.parametrize(
    "task, extra",
    [
        ("speaker_id", ["--utt2spk", "utt2spk", "--speaker-choices", "3"]),
        (
            "longform_wer",
            [
                "--text",
                "text",
                "--min-reference-words",
                "20",
                "--concatenate",
                "2",
                "--max-audio-seconds",
                "30",
            ],
        ),
        ("multispeaker_cpwer", ["--text", "text", "--utt2spk", "utt2spk"]),
    ],
)
def test_cli_prepares_each_task(tmp_path, monkeypatch, task, extra):
    wav, _, _ = corpus(tmp_path, per_speaker=2, sentences=2)
    arguments = [
        "tasks",
        task,
        "--wav-scp",
        str(wav),
        "--output",
        str(tmp_path / f"{task}.jsonl"),
    ]
    for item in extra:
        arguments.append(str(tmp_path / item) if item in ("text", "utt2spk") else item)
    monkeypatch.setattr(sys, "argv", arguments)
    tasks.main()
    rows = [
        json.loads(line)
        for line in (tmp_path / f"{task}.jsonl").read_text().splitlines()
    ]
    assert rows and all(row["reward_task"] == task for row in rows)


@pytest.mark.parametrize(
    "task, missing",
    [
        ("speaker_id", "--utt2spk"),
        ("longform_wer", "--text"),
        ("multispeaker_cpwer", "--text"),
    ],
)
def test_cli_requires_each_task_supervision(tmp_path, monkeypatch, task, missing):
    wav, _, _ = corpus(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tasks",
            task,
            "--wav-scp",
            str(wav),
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )
    with pytest.raises(SystemExit):
        tasks.main()
    assert not (tmp_path / "out.jsonl").exists()


def test_the_default_candidate_draw_does_not_leak_the_answer(tmp_path):
    """The candidate list must not become a function of the label.

    Keying the draw per speaker gives every utterance of one speaker the same
    list in the same order. That is harmless while training and heldout hold
    different speakers, but a closed-set split that keeps one speaker
    inventory and divides it by utterance then lets a model learn "this list
    means that label" and never listen to the audio. Measured on LibriSpeech
    dev-clean, 2674 records collapsed to 40 distinct lists with an injective
    list-to-label map. So the default keys per utterance.
    """
    wav, _, utt2spk = corpus(tmp_path, per_speaker=4)
    records = prepare_speaker_id_manifest(
        wav, utt2spk, tmp_path / "safe.jsonl", choices=4, seed=42
    )
    audit = json.loads((tmp_path / "safe.jsonl.audit.json").read_text())
    assert audit["choice_key"] == "utterance", "the safe mode must be the default"

    by_speaker = {}
    for record in records:
        by_speaker.setdefault(record["speaker"], set()).add(
            tuple(record["speaker_choices"])
        )
    # Every speaker has several utterances, and their lists must differ.
    assert all(len(lists) > 1 for lists in by_speaker.values()), by_speaker
    # And the list no longer determines the label.
    label_of = {}
    for record in records:
        label_of.setdefault(tuple(record["speaker_choices"]), set()).add(
            record["speaker"]
        )
    assert any(len(labels) > 1 for labels in label_of.values()), (
        "at least one candidate list must be shared by two different speakers, "
        "otherwise the list still identifies the answer"
    )
    speaker_id_reward.validate_sources(records)


def test_the_leaky_mode_is_still_reachable_but_must_be_asked_for(tmp_path):
    wav, _, utt2spk = corpus(tmp_path, per_speaker=4)
    records = prepare_speaker_id_manifest(
        wav, utt2spk, tmp_path / "legacy.jsonl", choices=4, seed=42,
        choice_key="speaker",
    )
    by_speaker = {}
    for record in records:
        by_speaker.setdefault(record["speaker"], set()).add(
            tuple(record["speaker_choices"])
        )
    # The old behaviour: one list per speaker, order included.
    assert all(len(lists) == 1 for lists in by_speaker.values())
    assert json.loads(
        (tmp_path / "legacy.jsonl.audit.json").read_text()
    )["choice_key"] == "speaker"


def test_every_registered_task_has_its_own_command():
    """A task module is one file plus one TASKS line; the CLI lists it."""
    import importlib

    from espnet2.rl.tasks import TASKS

    usage = tasks.parser().format_help()
    for name, path in TASKS.items():
        module = importlib.import_module(path)
        assert callable(module.add_arguments) and callable(module.run), name
        assert name in usage, name
