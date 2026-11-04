"""The torch/torchaudio skew warning must not break an interpreter's imports.

The RL rollout and export workers run in their own interpreter, which installs
torch from a local build and no torchaudio. ``importlib.metadata.version``
reads a version back as ``None`` there rather than raising
``PackageNotFoundError``, because the distribution's metadata carries no
Version field. The guard used to call ``.split()`` on that ``None``, so every
subprocess that imported ``espnet2`` with such an interpreter died at import
time, before running any of its own code.
"""

from importlib import metadata

import pytest

import espnet2


@pytest.mark.parametrize(
    "versions",
    [
        {"torch": None, "torchaudio": None},
        {"torch": "2.11.0+cu130", "torchaudio": None},
        {"torch": None, "torchaudio": "2.11.0"},
        {"torch": "", "torchaudio": ""},
    ],
)
def test_a_version_without_a_number_warns_about_nothing(monkeypatch, versions):
    import warnings

    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The call must return, not raise: this is what the rollout and export
        # interpreters execute on every start.
        espnet2._warn_if_torch_torchaudio_skewed()
    assert not [
        item for item in caught if issubclass(item.category, RuntimeWarning)
    ], "an unreadable version is not something to warn about"


def test_a_real_skew_still_warns(monkeypatch):
    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: {"torch": "2.11.0", "torchaudio": "2.1.0"}[name],
    )
    with pytest.warns(RuntimeWarning, match="matched pair"):
        espnet2._warn_if_torch_torchaudio_skewed()


def test_a_matched_pair_and_the_pinned_torchaudio_stay_quiet(monkeypatch):
    import warnings

    for pair in (("2.11.0", "2.11.0"), ("2.13.0+cu130", "2.11.0")):
        monkeypatch.setattr(
            metadata,
            "version",
            lambda name, pair=pair: dict(zip(("torch", "torchaudio"), pair))[name],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            espnet2._warn_if_torch_torchaudio_skewed()
        assert not [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ], pair


def test_an_absent_distribution_is_still_handled(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    espnet2._warn_if_torch_torchaudio_skewed()
