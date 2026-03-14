from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import Dict, List, Optional, Tuple

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


def _compact_text_with_index_map(text: str) -> Tuple[str, List[int]]:
    """Return ``text`` without whitespace plus an index map back to the original string.

    Corrected lyric segments are often tokenised with spaces between words or even between
    kanji characters (for example ``一 緒`` or ``未 来``). Ruby directives, however, refer
    to the visible base text without those inserted spaces. Compacting whitespace lets us
    match multi-character annotations against those segmented outputs while still mapping
    the match back to the original rendered text for ASS positioning.
    """
    compact_chars: List[str] = []
    index_map: List[int] = []

    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        index_map.append(index)

    return "".join(compact_chars), index_map


def _find_unused_match(compact_text: str, needle: str, used_spans: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Find the next non-overlapping occurrence of ``needle`` inside ``compact_text``."""
    search_from = 0

    while True:
        match_start = compact_text.find(needle, search_from)
        if match_start == -1:
            return None

        match_end = match_start + len(needle)
        overlaps_existing = any(match_start < used_end and match_end > used_start for used_start, used_end in used_spans)
        if not overlaps_existing:
            return match_start, match_end

        search_from = match_start + 1


def resolve_ruby_annotations_for_segments(
    segments: List[LyricsSegment],
    annotations: List[RubyAnnotation],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[ResolvedRubyAnnotation]]:
    """Resolve ordered ruby annotations onto output segments in chronological order.

    Ruby indices are only reliable at the *segment* level. Within a single line they may be
    out of visual order (for example ``見`` can be indexed before ``君`` inside ``君を見つめた``),
    and corrected output may insert spaces between tokens (for example ``一 緒``). To keep the
    mapping stable we therefore:

    - walk segments monotonically in song order;
    - allow matches anywhere inside the current segment, regardless of prior character order;
    - ignore inserted whitespace while searching;
    - leave the segment cursor unchanged when one annotation is unresolved so later ruby can
      still resolve instead of being poisoned by a single miss.
    """
    if not annotations or not segments:
        return {}

    logger = logger or logging.getLogger(__name__)
    resolved: Dict[int, List[ResolvedRubyAnnotation]] = {}
    compact_segments = [_compact_text_with_index_map(segment.text) for segment in segments]
    used_spans: Dict[int, List[Tuple[int, int]]] = {index: [] for index in range(len(segments))}
    segment_index = 0
    unresolved_count = 0

    for annotation in annotations:
        normalized_base_text = "".join(char for char in annotation.base_text if not char.isspace())
        if not normalized_base_text:
            unresolved_count += 1
            logger.warning(
                "Could not resolve ruby annotation @Ruby%s=%s,%s onto output segments",
                annotation.index,
                annotation.base_text,
                annotation.ruby_text,
            )
            continue

        found = False

        for candidate_index in range(segment_index, len(segments)):
            compact_text, index_map = compact_segments[candidate_index]
            if not compact_text:
                continue

            compact_match = _find_unused_match(compact_text, normalized_base_text, used_spans[candidate_index])
            if compact_match is None:
                continue

            compact_start, compact_end = compact_match
            used_spans[candidate_index].append((compact_start, compact_end))

            start_char = index_map[compact_start]
            end_char = index_map[compact_end - 1] + 1
            resolved.setdefault(candidate_index, []).append(
                ResolvedRubyAnnotation(
                    index=annotation.index,
                    base_text=annotation.base_text,
                    ruby_text=annotation.ruby_text,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            segment_index = candidate_index
            found = True
            break

        if not found:
            unresolved_count += 1
            logger.warning(
                "Could not resolve ruby annotation @Ruby%s=%s,%s onto output segments",
                annotation.index,
                annotation.base_text,
                annotation.ruby_text,
            )

    for annotations_for_segment in resolved.values():
        annotations_for_segment.sort(key=lambda item: (item.start_char, item.end_char, item.index))

    logger.info(
        "Resolved %d/%d ruby annotations onto output segments",
        len(annotations) - unresolved_count,
        len(annotations),
    )

    return resolved
