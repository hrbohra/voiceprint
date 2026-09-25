"""The one data model every loader produces and every stage consumes.

A corpus is a list of turns. A turn is one message (or one paragraph of prose) by one speaker,
optionally inside a conversation (`doc_id`) with a position (`turn_id`) and a pointer to the turn
it answers (`reply_to`). Keeping every format in this shape is what lets one pipeline read
Instagram DMs, support tweets, email and novels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


@dataclass
class Turn:
    doc_id: str
    turn_id: int
    speaker: str
    text: str
    timestamp: datetime | None = None
    reply_to: int | None = None  # turn_id this answers, within the same doc
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.turn_id}"


@dataclass
class Corpus:
    turns: list[Turn]
    name: str = "corpus"

    def __len__(self) -> int:
        return len(self.turns)

    def speakers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.turns:
            counts[t.speaker] = counts.get(t.speaker, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def docs(self) -> dict[str, list[Turn]]:
        out: dict[str, list[Turn]] = {}
        for t in self.turns:
            out.setdefault(t.doc_id, []).append(t)
        for v in out.values():
            v.sort(key=lambda x: x.turn_id)
        return out

    def by_speaker(self, speakers: Iterable[str]) -> list[Turn]:
        wanted = set(speakers)
        return [t for t in self.turns if t.speaker in wanted]

    def previous_turn(self) -> dict[str, Turn | None]:
        """For every turn, the turn it answers: explicit reply_to, else the previous turn by
        someone else in the same conversation. Context for dialogue acts and accommodation."""
        prev: dict[str, Turn | None] = {}
        for turns in self.docs().values():
            by_id = {t.turn_id: t for t in turns}
            last_other: dict[str, Turn] = {}
            last_any: Turn | None = None
            for t in turns:
                if t.reply_to is not None and t.reply_to in by_id:
                    prev[t.key] = by_id[t.reply_to]
                else:
                    cand = last_any if last_any is not None and last_any.speaker != t.speaker else None
                    if cand is None:
                        # the most recent turn by anyone else
                        others = [v for k, v in last_other.items() if k != t.speaker]
                        cand = max(others, key=lambda x: x.turn_id) if others else None
                    prev[t.key] = cand
                last_other[t.speaker] = t
                last_any = t
        return prev
