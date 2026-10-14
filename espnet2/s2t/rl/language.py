# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint-bound language aliases matching espnet/vLLM without importing it."""

_ISO_LANGUAGE_PAIRS = [
    "en:eng",
    "zh:zho",
    "de:deu",
    "es:spa",
    "ru:rus",
    "ko:kor",
    "fr:fra",
    "ja:jpn",
    "pt:por",
    "tr:tur",
    "pl:pol",
    "ca:cat",
    "nl:nld",
    "ar:ara",
    "sv:swe",
    "it:ita",
    "id:ind",
    "hi:hin",
    "fi:fin",
    "vi:vie",
    "he:heb",
    "uk:ukr",
    "el:ell",
    "ms:msa",
    "cs:ces",
    "ro:ron",
    "da:dan",
    "hu:hun",
    "ta:tam",
    "no:nor",
    "th:tha",
    "ur:urd",
    "hr:hrv",
    "bg:bul",
    "lt:lit",
    "la:lat",
    "mi:mri",
    "ml:mal",
    "cy:cym",
    "sk:slk",
    "te:tel",
    "fa:fas",
    "lv:lav",
    "bn:ben",
    "sr:srp",
    "az:aze",
    "sl:slv",
    "kn:kan",
    "et:est",
    "mk:mkd",
    "br:bre",
    "eu:eus",
    "is:isl",
    "hy:hye",
    "ne:nep",
    "mn:mon",
    "bs:bos",
    "kk:kaz",
    "sq:sqi",
    "sw:swa",
    "gl:glg",
    "mr:mar",
    "pa:pan",
    "si:sin",
    "km:khm",
    "sn:sna",
    "yo:yor",
    "so:som",
    "af:afr",
    "oc:oci",
    "ka:kat",
    "be:bel",
    "tg:tgk",
    "sd:snd",
    "gu:guj",
    "am:amh",
    "yi:yid",
    "lo:lao",
    "uz:uzb",
    "fo:fao",
    "ht:hat",
    "ps:pus",
    "tk:tuk",
    "nn:nno",
    "mt:mlt",
    "sa:san",
    "lb:ltz",
    "my:mya",
    "bo:bod",
    "tl:tgl",
    "mg:mlg",
    "as:asm",
    "tt:tat",
    "ln:lin",
    "ha:hau",
    "ba:bak",
    "su:sun",
    "jw:jav",
]


def resolve_owsm_language_symbol(tokens, language, *, translation=False):
    """Resolve an ISO code/name to one symbol actually defined by the checkpoint.

    Exact symbols take precedence. A broad language name with multiple available
    symbols is ambiguous and must be replaced with an explicit checkpoint code.
    """
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    value = language.strip().lower().removeprefix("<").removesuffix(">")
    if value.startswith("st_"):
        value = value[3:]
        translation = True
    prefix = "st_" if translation else ""
    exact = f"<{prefix}{value}>"
    if exact in tokens:
        return exact
    names = {code: name for code, name in LANGUAGES.items()}
    groups = [
        {short, full, names.get(short, short)}
        for short, full in (pair.split(":") for pair in _ISO_LANGUAGE_PAIRS)
    ]
    # OWSM v3+ distinguishes Mandarin from other Chinese languages.
    groups[next(i for i, group in enumerate(groups) if "zh" in group)].add("cmn")
    groups.extend(({"haw", "hawaiian"}, {"yue", "cantonese"}))
    aliases = next((group for group in groups if value in group), {value})
    matches = sorted(
        f"<{prefix}{alias}>" for alias in aliases if f"<{prefix}{alias}>" in tokens
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous OWSM language {language!r}: {matches}")
    raise ValueError(f"Checkpoint does not define OWSM language {language!r}")
