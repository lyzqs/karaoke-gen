from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import Dict, List, Optional

from karaoke_gen.lyrics_transcriber.types import LyricsSegment


_RUBY_DIRECTIVE_RE = re.compile(r"^@Ruby(?P<index>\d+)=(?P<body>.+)$")


@dataclass(frozen=True)
class RubyAnnotation:
    """A ruby/furigana directive parsed from an input lyrics file."""

    index: int
    base_text: str
    ruby_text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RubyAnnotation":
        return cls(
            index=int(data["index"]),
            base_text=str(data["base_text"]),
            ruby_text=str(data["ruby_text"]),
        )


@dataclass(frozen=True)
class ResolvedRubyAnnotation:
    """A ruby annotation resolved onto a specific output segment."""

    index: int
    base_text: str
    ruby_text: str
    start_char: int
    end_char: int


def parse_ruby_annotations_from_lrc_text(text: str) -> List[RubyAnnotation]:
    """Parse MidiCo-style ``@RubyN=base,reading`` directives from raw LRC text."""
    annotations: List[RubyAnnotation] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _RUBY_DIRECTIVE_RE.match(line)
        if not match:
            continue

        body = match.group("body")
        base_text, separator, ruby_text = body.partition(",")
        if not separator:
            continue

        base_text = base_text.strip()
        ruby_text = ruby_text.strip()
        if not base_text or not ruby_text:
            continue

        annotations.append(
            RubyAnnotation(
                index=int(match.group("index")),
                base_text=base_text,
                ruby_text=ruby_text,
            )
        )

    return sorted(annotations, key=lambda item: item.index)


def load_ruby_annotations_from_lrc_file(filepath: str) -> List[RubyAnnotation]:
    """Load ruby annotations from an LRC file path."""
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as handle:
        return parse_ruby_annotations_from_lrc_text(handle.read())


def deserialize_ruby_annotations(data: Optional[List[Dict[str, object]]]) -> List[RubyAnnotation]:
    """Convert serialized provider metadata back into ruby annotation objects."""
    if not data:
        return []
    return [RubyAnnotation.from_dict(item) for item in data]


def resolve_ruby_annotations_for_segments(
    segments: List[LyricsSegment],
    annotations: List[RubyAnnotation],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[ResolvedRubyAnnotation]]:
    """Resolve ordered ruby annotations onto output segments in chronological order.

    The input LRC directives do not encode explicit line/character positions. In practice,
    the directives are already ordered by appearance in the song, so we resolve them by
    walking the output segments left-to-right and matching the next occurrence of each
    ``base_text``.
    """
    if not annotations or not segments:
        return {}

    logger = logger or logging.getLogger(__name__)
    resolved: Dict[int, List[ResolvedRubyAnnotation]] = {}
    segment_index = 0
    search_offset = 0

    for annotation in annotations:
        found = False

        while segment_index < len(segments):
            segment_text = segments[segment_index].text
            match_start = segment_text.find(annotation.base_text, search_offset)
            if match_start == -1:
                segment_index += 1
                search_offset = 0
                continue

            match_end = match_start + len(annotation.base_text)
            resolved.setdefault(segment_index, []).append(
                ResolvedRubyAnnotation(
                    index=annotation.index,
                    base_text=annotation.base_text,
                    ruby_text=annotation.ruby_text,
                    start_char=match_start,
                    end_char=match_end,
                )
            )
            search_offset = match_end
            found = True
            break

        if not found:
            logger.warning(
                "Could not resolve ruby annotation @Ruby%s=%s,%s onto output segments",
                annotation.index,
                annotation.base_text,
                annotation.ruby_text,
            )

    return resolved
