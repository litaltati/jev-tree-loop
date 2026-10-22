"""Is a company really named in the message? Checked by spelling, not by the model.

Jev knows the world: asked whether "Globex" means one of the companies of the
report, it may say yes to another company it associates with it — the very
substitution the answers must never make. So an entity Jev picked is kept only
when the user's own words hold something spelled like its name; one it picked
without such a word is dropped and counted as "a name the data does not have".
The other way round, a misspelled name Jev missed ("akmetelin") is recovered
when its spelling is close enough.

Only the user's turns are read: an earlier answer may itself have used the
wrong name.
"""

from __future__ import annotations

import difflib
import re

CLOSE = 0.8           # one or two letters off in a name of six or more
NEAR = 0.6            # alike enough to be a sibling's name, not a different one
ENDINGS = (1, 2, 3)   # an Azerbaijani or Russian ending: acmetelin, acmenetin, акметела

_CYRILLIC = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
     "u", "f", "kh", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))
_AZERBAIJANI = {"ə": "e", "ı": "i", "ö": "o", "ü": "u", "ş": "s", "ç": "c", "ğ": "g"}
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def sound(text: str) -> str:
    """A rough spelling key shared by Latin, Azerbaijani and Cyrillic writing:
    'Acmetel', 'akmetel' and 'Акметел' all come out close."""
    out = []
    for ch in text.lower():
        ch = _AZERBAIJANI.get(ch, ch)
        out.append(_CYRILLIC.get(ch, ch))
    key = "".join(out).replace("c", "k")
    return re.sub(r"(.)\1+", r"\1", key)        # double letters count once


def name_key(entity: str) -> str | None:
    """The part of an entity's name that can be matched by spelling: its first
    word of three letters or more. None when it has no such word ("m a")."""
    return next((sound(w) for w in entity.split() if len(w) >= 3 and not w.isdigit()), None)


def _score(word_key: str, key: str) -> float:
    """1.0 for the name itself (an ending of up to three letters allowed), else
    how alike the spellings are — the whole word, or the word cut to the length
    of the name when only an ending is left over. Short cut-down forms are never
    compared loosely: "kos" of "costs" would pass for any four-letter name."""
    forms = [word_key] + [word_key[:-cut] for cut in ENDINGS if len(word_key) - cut >= 3]
    if key in forms:
        return 1.0
    if len(key) < 4:
        return 0.0
    score = difflib.SequenceMatcher(None, word_key, key).ratio()
    if 0 <= len(word_key) - len(key) <= max(ENDINGS):
        score = max(score, difflib.SequenceMatcher(None, word_key[: len(key)], key).ratio())
    return score


def spelling(entities: list[str], words: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """For each entity: (own, sibling).

    own — the best score of a word that names it, each word counted only for the
    entity it is closest to, so "Acmenet" does not name Acmetel as well.
    sibling — how alike it is to a word that clearly names another entity of the
    data: Jev picking Acmetel for "Acmenet" is a slip between neighbours, while
    picking it for "Globex" or "Initech" — words that name nothing in the data — is
    the user asking about a company the data does not have.
    """
    keys = {e: name_key(e) for e in entities}
    own = {e: 0.0 for e in entities}
    sibling = {e: 0.0 for e in entities}
    for word in words:
        word_key = sound(word)
        scores = {e: _score(word_key, k) for e, k in keys.items() if k}
        if not scores:
            continue
        top = max(scores.values())
        for e, score in scores.items():
            if score == top:                      # siblings of one name ("x", "x int") tie
                own[e] = max(own[e], score)
            elif top >= CLOSE:
                sibling[e] = max(sibling[e], score)
    return own, sibling


def user_words(message: str, history: list[dict[str, str]] | None) -> list[str]:
    texts = [message] + [t["content"] for t in (history or []) if t.get("role") == "user" and t.get("content")]
    return [w for text in texts for w in _WORD.findall(text) if len(w) >= 3]


def reconcile(entities: list[str], jev_yes: dict[str, bool], words: list[str]) -> tuple[list[str], bool]:
    """(the entities to answer for, whether a name the data lacks was asked about).

    Kept: what Jev chose and the user wrote. Dropped quietly: what Jev chose
    because the user wrote a sibling's name ("Acmenet" — not Acmetel). Dropped
    as missing: what Jev chose for a word that names nothing in the data ("Globex").
    Recovered: a misspelled name, when nothing Jev found is left.
    """
    keys = {e: name_key(e) for e in entities}
    own, sibling = spelling(entities, words)
    kept, missing = [], False
    for e in entities:
        if not jev_yes[e]:
            continue
        if keys[e] is None or own[e] >= CLOSE:
            kept.append(e)
        elif sibling[e] < NEAR:
            missing = True
    if not kept:
        # Jev found no company, or every one it found was dropped: the user's own
        # spelling decides — and a name recovered that way is not a missing one.
        kept = [e for e in entities if keys[e] and len(keys[e]) >= 4 and own[e] >= CLOSE]
        if kept:
            missing = False
    return _one_of_siblings(kept, words), missing


def _one_of_siblings(kept: list[str], words: list[str]) -> list[str]:
    """Of names that share their first word ("acmetel", "acmetel int"),
    keep the one whose other words the user wrote — or the plain one."""
    keys = {e: name_key(e) for e in kept}
    written = [sound(w) for w in words]
    out = []
    for e in kept:
        siblings = [o for o in kept if keys[o] and keys[o] == keys[e]]
        if len(siblings) < 2:
            out.append(e)
            continue
        def extra_written(entity: str) -> bool:
            extra = [sound(t) for t in entity.split()[1:] if len(t) >= 2]
            return bool(extra) and all(any(w.startswith(t) for w in written) for t in extra)
        with_extra = [o for o in siblings if extra_written(o)]
        chosen = with_extra or [o for o in siblings if len(o.split()) == 1] or siblings
        if e in chosen:
            out.append(e)
    return out
