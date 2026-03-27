from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from karaoke_gen.video_generator import _find_cjk_font

try:
    from fugashi import Tagger
except ImportError:  # pragma: no cover
    Tagger = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2})(?:[.:](\d{2,3}))?\]")
RUBY_DIRECTIVE_RE = re.compile(r"^@Ruby(?P<index>\d+)=(?P<base>[^,]+),(?P<reading>.+)$")
KANJI_RE = re.compile(r"[一-龯々]")
KANJI_ONLY_RE = re.compile(r"^[一-龯々]+$")
FALLBACK_TOKEN_RE = re.compile(r"[一-龯々]+[ぁ-ゖ]*|[ぁ-ゖ]+|[ァ-ヺー]+|[A-Za-z0-9']+|[^\s]")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
SMALL_KANA = set("ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮ")
READING_OVERRIDES = {
    "君": "きみ",
    "私": "わたし",
}
PARTICLE_SURFACES = {"を", "が", "は", "に", "へ", "で", "と", "も", "の", "から", "まで", "だけ", "って"}

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
}

TXT_MIN_LINE_DURATION_MS = 600
TXT_TARGET_LINE_DURATION_MS = 1_800
TXT_LEAD_IN_RATIO = 0.12
TXT_LEAD_OUT_RATIO = 0.10
TXT_MIN_LEAD_IN_MS = 3_000
TXT_MIN_LEAD_OUT_MS = 2_000
TXT_MAX_LEAD_IN_MS = 25_000
TXT_MAX_LEAD_OUT_MS = 20_000

CURRENT_ACTIVE_FILL_ASS = "&H0030C7FF&"
CURRENT_INACTIVE_FILL_ASS = "&H00F6F6FF&"
UPCOMING_FILL_ASS = "&H00E8F0FF&"
TEXT_OUTLINE_ASS = "&H0010131D&"
ALIGNMENT_SAMPLE_RATE = 16_000
ALIGNMENT_PRE_ROLL_MS = 900
ALIGNMENT_POST_ROLL_MS = 260
ALIGNMENT_DEVICE = "cpu"
ALIGNMENT_COMPUTE_TYPE = "int8"
ALIGNMENT_MODEL_SIZE = "small"
MIN_ALIGNMENT_CUE_MS = 40


@dataclass(frozen=True)
class TokenPiece:
    surface: str
    reading: str
    pos1: str = ""
    pos2: str = ""
    has_kanji: bool = False
    punctuation: bool = False


@dataclass(frozen=True)
class TimedToken(TokenPiece):
    start_ms: int = 0
    end_ms: int = 0


@dataclass(frozen=True)
class RubyGroup:
    surface: str
    reading: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TimedLine:
    start_ms: int
    end_ms: int
    text: str
    tokens: list[TimedToken]
    ruby_groups: list[RubyGroup]


@dataclass(frozen=True)
class RenderResult:
    video_path: Path
    ass_path: Path
    timeline_path: Path
    duration_ms: int
    txt_path: Path
    lrc_path: Path


@dataclass(frozen=True)
class FontSet:
    main: ImageFont.FreeTypeFont
    ruby: ImageFont.FreeTypeFont


@dataclass(frozen=True)
class LayoutMetrics:
    left: int
    right: int
    width: int
    token_bounds: list[tuple[int, int]]
    char_bounds: list[tuple[int, int]]


@dataclass(frozen=True)
class RubyDirective:
    index: int
    base_text: str
    ruby_text: str


@dataclass(frozen=True)
class ResolvedRubyDirective:
    index: int
    base_text: str
    ruby_text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class AudioWordCue:
    surface: str
    start_ms: int
    end_ms: int


_ALIGNMENT_MODEL: Optional["WhisperModel"] = None


class JapaneseTokenizer:
    def __init__(self) -> None:
        self._tagger = Tagger() if Tagger is not None else None

    def tokenize(self, text: str) -> list[TokenPiece]:
        if self._tagger is not None:
            pieces = self._tokenize_with_fugashi(text)
        else:  # pragma: no cover
            pieces = self._tokenize_fallback(text)
        return self._merge_numeric_compounds(pieces)

    def _tokenize_with_fugashi(self, text: str) -> list[TokenPiece]:
        words = list(self._tagger(text))
        pieces: list[TokenPiece] = []

        for index, word in enumerate(words):
            surface = word.surface
            feature = word.feature
            next_surface = words[index + 1].surface if index + 1 < len(words) else ""
            reading = _reading_override(
                surface=surface,
                base_reading=_kata_to_hiragana(
                    getattr(feature, "kana", None)
                    or getattr(feature, "kanaBase", None)
                    or surface
                ),
                next_surface=next_surface,
            )
            pieces.append(
                TokenPiece(
                    surface=surface,
                    reading=reading,
                    pos1=getattr(feature, "pos1", "") or "",
                    pos2=getattr(feature, "pos2", "") or "",
                    has_kanji=bool(KANJI_RE.search(surface)),
                    punctuation=_is_punctuation(surface),
                )
            )

        return pieces

    def _tokenize_fallback(self, text: str) -> list[TokenPiece]:  # pragma: no cover
        pieces: list[TokenPiece] = []
        for surface in FALLBACK_TOKEN_RE.findall(text):
            pieces.append(
                TokenPiece(
                    surface=surface,
                    reading=_kata_to_hiragana(surface),
                    has_kanji=bool(KANJI_RE.search(surface)),
                    punctuation=_is_punctuation(surface),
                )
            )
        return pieces

    def _merge_numeric_compounds(self, pieces: list[TokenPiece]) -> list[TokenPiece]:
        merged: list[TokenPiece] = []
        for piece in pieces:
            if merged and _can_merge_numeric_tokens(merged[-1], piece):
                previous = merged.pop()
                merged.append(
                    TokenPiece(
                        surface=previous.surface + piece.surface,
                        reading=previous.reading + piece.reading,
                        pos1=previous.pos1,
                        pos2=previous.pos2,
                        has_kanji=True,
                        punctuation=False,
                    )
                )
            else:
                merged.append(piece)
        return merged


def parse_lrc_timed_lines(raw_text: str, tokenizer: Optional[JapaneseTokenizer] = None) -> list[TimedLine]:
    tokenizer = tokenizer or JapaneseTokenizer()
    parsed_lines: list[tuple[int, str]] = []
    ruby_directives = _parse_ruby_directives(raw_text)

    for raw_line in raw_text.splitlines():
        matches = list(TIMESTAMP_RE.finditer(raw_line))
        if not matches:
            continue
        lyric_text = TIMESTAMP_RE.sub("", raw_line).strip()
        if not lyric_text:
            continue
        for match in matches:
            parsed_lines.append((_parse_timestamp(match), lyric_text))

    if not parsed_lines:
        return []

    resolved_ruby = _resolve_ruby_directives_for_line_texts([text for _, text in parsed_lines], ruby_directives)
    timed_lines: list[TimedLine] = []

    for index, (start_ms, text) in enumerate(parsed_lines):
        if index + 1 < len(parsed_lines):
            end_ms = parsed_lines[index + 1][0]
        else:
            end_ms = start_ms + max(2_500, len(text) * 240)
        end_ms = max(end_ms, start_ms + TXT_MIN_LINE_DURATION_MS)

        token_pieces = tokenizer.tokenize(text)
        timed_tokens = _assign_weighted_token_timing(token_pieces, start_ms, end_ms)
        explicit_groups = _ruby_groups_from_resolved_directives(
            text=text,
            tokens=timed_tokens,
            resolved_directives=resolved_ruby.get(index, []),
        )
        ruby_groups = explicit_groups or _build_ruby_groups(timed_tokens)
        timed_lines.append(
            TimedLine(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                tokens=timed_tokens,
                ruby_groups=ruby_groups,
            )
        )

    return timed_lines


def parse_txt_timed_lines(
    raw_text: str,
    duration_ms: int,
    tokenizer: Optional[JapaneseTokenizer] = None,
    timing_bridge_lines: Optional[list[TimedLine]] = None,
) -> list[TimedLine]:
    if duration_ms <= 0:
        raise ValueError("TXT timing derivation requires a positive duration")

    tokenizer = tokenizer or JapaneseTokenizer()
    lyric_lines = _extract_lyric_text_lines(raw_text)
    if not lyric_lines:
        return []

    if timing_bridge_lines:
        bridged_lines = _retime_txt_lines_from_bridge(lyric_lines, timing_bridge_lines, tokenizer)
        if bridged_lines:
            return bridged_lines

    timed_lines: list[TimedLine] = []
    tokenized_lines = [(text, tokenizer.tokenize(text)) for text in lyric_lines]
    line_weights = [max(1, _line_weight(tokens)) for _, tokens in tokenized_lines]

    content_start_ms, content_end_ms = _derive_txt_timing_window(duration_ms, len(tokenized_lines))
    content_duration_ms = max(content_end_ms - content_start_ms, len(tokenized_lines) * TXT_MIN_LINE_DURATION_MS)
    total_weight = max(sum(line_weights), 1)

    line_starts = [content_start_ms]
    accumulated_weight = 0
    for index, weight in enumerate(line_weights[:-1], start=1):
        accumulated_weight += weight
        proportional_start = content_start_ms + round(content_duration_ms * accumulated_weight / total_weight)
        minimum_start = line_starts[-1] + TXT_MIN_LINE_DURATION_MS
        remaining_lines = len(tokenized_lines) - index
        maximum_start = content_end_ms - remaining_lines * TXT_MIN_LINE_DURATION_MS
        line_starts.append(min(max(proportional_start, minimum_start), maximum_start))

    line_boundaries = [*line_starts, content_end_ms]
    for index, ((text, token_pieces), start_ms, end_ms) in enumerate(
        zip(tokenized_lines, line_boundaries, line_boundaries[1:])
    ):
        if index == len(tokenized_lines) - 1:
            end_ms = content_end_ms
        end_ms = max(end_ms, start_ms + TXT_MIN_LINE_DURATION_MS)
        timed_tokens = _assign_weighted_token_timing(token_pieces, start_ms, end_ms)
        timed_lines.append(
            TimedLine(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                tokens=timed_tokens,
                ruby_groups=_build_ruby_groups(timed_tokens),
            )
        )

    return timed_lines


def detect_font_path() -> Optional[Path]:
    direct = _find_cjk_font()
    if direct:
        return Path(direct)

    try:
        result = subprocess.run(
            ["fc-match", "--format=%{file}", "Noto Sans CJK JP"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: no cover
        return None

    candidate = result.stdout.strip()
    if candidate and Path(candidate).exists():
        return Path(candidate)
    return None


def load_font_set(font_path: Optional[Path]) -> FontSet:
    if font_path is None or not font_path.exists():
        raise FileNotFoundError("A CJK-capable font is required for local JP rendering")

    return FontSet(
        main=ImageFont.truetype(str(font_path), size=80),
        ruby=ImageFont.truetype(str(font_path), size=36),
    )


def build_ass_document(
    timed_lines: list[TimedLine],
    resolution: str,
    fonts: FontSet,
    font_family: str,
) -> str:
    width, height = RESOLUTION_MAP[resolution]
    measure_image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(measure_image)

    main_font_size = 80
    ruby_font_size = 36
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: CurrentLine,{font_family},{main_font_size},{CURRENT_ACTIVE_FILL_ASS},{CURRENT_INACTIVE_FILL_ASS},{TEXT_OUTLINE_ASS},&H50000000,1,0,0,0,100,100,0,0,1,4.4,0.7,8,64,64,0,1",
        f"Style: UpcomingLine,{font_family},{main_font_size},{UPCOMING_FILL_ASS},{UPCOMING_FILL_ASS},{TEXT_OUTLINE_ASS},&H50000000,1,0,0,0,100,100,0,0,1,4.4,0.7,8,64,64,0,1",
        f"Style: RubyCurrent,{font_family},{ruby_font_size},&H00FFF7EB,&H00FFF7EB,{TEXT_OUTLINE_ASS},&H46000000,0,0,0,0,100,100,0,0,1,2.4,0.4,8,64,64,0,1",
        f"Style: RubyUpcoming,{font_family},{ruby_font_size},&H00FFF7EB,&H00FFF7EB,{TEXT_OUTLINE_ASS},&H46000000,0,0,0,0,100,100,0,0,1,2.4,0.4,8,64,64,0,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    for index, timed_line in enumerate(timed_lines):
        current_layout = _layout_line(
            draw=draw,
            tokens=timed_line.tokens,
            font=fonts.main,
            canvas_width=width,
            line_index=index,
        )
        current_alignment, current_anchor_x, current_y = _line_anchor(index, width, height)
        current_text = _build_karaoke_text(timed_line.tokens, timed_line.start_ms)
        lines.append(
            _format_dialogue(
                layer=0,
                start_ms=timed_line.start_ms,
                end_ms=timed_line.end_ms,
                style="CurrentLine",
                text=(
                    f"{{\\an{current_alignment}\\pos({current_anchor_x},{current_y})\\q2}}"
                    f"{{\\1c{CURRENT_ACTIVE_FILL_ASS}\\2c{CURRENT_INACTIVE_FILL_ASS}\\3c{TEXT_OUTLINE_ASS}}}{current_text}"
                ),
            )
        )
        lines.extend(
            _build_ruby_dialogue_lines(
                line=timed_line,
                layout=current_layout,
                style="RubyCurrent",
                layer=2,
                start_ms=timed_line.start_ms,
                end_ms=timed_line.end_ms,
                ruby_y=_ruby_y(current_y),
            )
        )

        if index + 1 >= len(timed_lines):
            continue

        upcoming_line = timed_lines[index + 1]
        upcoming_layout = _layout_line(
            draw=draw,
            tokens=upcoming_line.tokens,
            font=fonts.main,
            canvas_width=width,
            line_index=index + 1,
        )
        upcoming_alignment, upcoming_anchor_x, upcoming_y = _line_anchor(index + 1, width, height)
        lines.append(
            _format_dialogue(
                layer=1,
                start_ms=timed_line.start_ms,
                end_ms=timed_line.end_ms,
                style="UpcomingLine",
                text=(
                    f"{{\\an{upcoming_alignment}\\pos({upcoming_anchor_x},{upcoming_y})\\q2}}"
                    f"{{\\1c{UPCOMING_FILL_ASS}\\3c{TEXT_OUTLINE_ASS}}}{_escape_ass(upcoming_line.text)}"
                ),
            )
        )
        lines.extend(
            _build_ruby_dialogue_lines(
                line=upcoming_line,
                layout=upcoming_layout,
                style="RubyUpcoming",
                layer=3,
                start_ms=timed_line.start_ms,
                end_ms=timed_line.end_ms,
                ruby_y=_ruby_y(upcoming_y),
            )
        )

    return "\n".join(lines) + "\n"


def write_ass_subtitles(
    timed_lines: list[TimedLine],
    output_path: Path,
    resolution: str,
    fonts: FontSet,
    font_family: str,
) -> Path:
    output_path.write_text(
        build_ass_document(
            timed_lines=timed_lines,
            resolution=resolution,
            fonts=fonts,
            font_family=font_family,
        ),
        encoding="utf-8",
    )
    return output_path


def write_timeline_debug(timed_lines: list[TimedLine], output_path: Path) -> Path:
    output_path.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "start_ms": line.start_ms,
                        "end_ms": line.end_ms,
                        "text": line.text,
                        "tokens": [
                            {
                                "surface": token.surface,
                                "reading": token.reading,
                                "start_ms": token.start_ms,
                                "end_ms": token.end_ms,
                            }
                            for token in line.tokens
                        ],
                        "ruby_groups": [
                            {
                                "surface": group.surface,
                                "reading": group.reading,
                                "start_ms": group.start_ms,
                                "end_ms": group.end_ms,
                            }
                            for group in line.ruby_groups
                        ],
                    }
                    for line in timed_lines
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def build_lrc_document(timed_lines: list[TimedLine]) -> str:
    return "\n".join(f"{_format_lrc_timestamp(line.start_ms)}{line.text}" for line in timed_lines) + "\n"


def build_txt_document(timed_lines: list[TimedLine]) -> str:
    return "\n".join(line.text for line in timed_lines) + "\n"


def write_txt_sidecar(timed_lines: list[TimedLine], output_path: Path) -> Path:
    output_path.write_text(build_txt_document(timed_lines), encoding="utf-8")
    return output_path


def write_lrc_sidecar(timed_lines: list[TimedLine], output_path: Path) -> Path:
    output_path.write_text(build_lrc_document(timed_lines), encoding="utf-8")
    return output_path


def _align_timed_lines_to_audio(
    timed_lines: list[TimedLine],
    audio_path: Path,
    tokenizer: Optional[JapaneseTokenizer] = None,
) -> list[TimedLine]:
    if not timed_lines:
        return []
    if WhisperModel is None:  # pragma: no cover
        raise RuntimeError("faster-whisper is required for local JP audio alignment")

    tokenizer = tokenizer or JapaneseTokenizer()
    audio = _load_alignment_audio(audio_path)
    model = _get_alignment_model()

    aligned_payloads: list[tuple[TimedLine, list[TimedToken]]] = []
    for line in timed_lines:
        aligned_tokens = _align_line_tokens(
            model=model,
            audio=audio,
            line=line,
            tokenizer=tokenizer,
        )
        if not aligned_tokens:
            raise RuntimeError(f"Failed to derive aligned timings for line: {line.text}")

        if aligned_payloads:
            previous_tokens = aligned_payloads[-1][1]
            overlap_ms = previous_tokens[-1].end_ms - aligned_tokens[0].start_ms
            if overlap_ms > 0:
                aligned_tokens = _shift_tokens(aligned_tokens, overlap_ms)

        aligned_payloads.append((line, aligned_tokens))

    aligned_lines: list[TimedLine] = []
    aligned_starts = [tokens[0].start_ms for _, tokens in aligned_payloads]
    for index, (line, aligned_tokens) in enumerate(aligned_payloads):
        aligned_start_ms = aligned_starts[index]
        if index + 1 < len(aligned_starts):
            aligned_end_ms = aligned_starts[index + 1]
        else:
            aligned_end_ms = max(line.end_ms, aligned_tokens[-1].end_ms)

        aligned_lines.append(
            TimedLine(
                start_ms=aligned_start_ms,
                end_ms=aligned_end_ms,
                text=line.text,
                tokens=aligned_tokens,
                ruby_groups=_remap_ruby_groups(line.text, line.ruby_groups, aligned_tokens),
            )
        )

    return aligned_lines


def _get_alignment_model() -> "WhisperModel":
    global _ALIGNMENT_MODEL
    if _ALIGNMENT_MODEL is None:
        _ALIGNMENT_MODEL = WhisperModel(
            ALIGNMENT_MODEL_SIZE,
            device=ALIGNMENT_DEVICE,
            compute_type=ALIGNMENT_COMPUTE_TYPE,
        )
    return _ALIGNMENT_MODEL


def _load_alignment_audio(audio_path: Path) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for local JP audio alignment")

    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(ALIGNMENT_SAMPLE_RATE),
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"Failed to decode alignment audio from {audio_path}")

    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _align_line_tokens(
    model: "WhisperModel",
    audio: np.ndarray,
    line: TimedLine,
    tokenizer: JapaneseTokenizer,
) -> list[TimedToken]:
    cues = _transcribe_line_cues(model=model, audio=audio, line=line)
    matched_cues = _stabilize_matched_cues(
        _match_cues_to_line_text(line.text, cues, tokenizer),
        line_end_ms=line.end_ms,
    )

    if not matched_cues:
        raise RuntimeError(f"No aligned words matched lyric text: {line.text}")

    timed_tokens: list[TimedToken] = []
    for cue in matched_cues:
        piece = _token_piece_from_surface(cue.surface, tokenizer)
        start_ms = cue.start_ms
        end_ms = max(start_ms + 1, cue.end_ms)
        timed_tokens.append(
            TimedToken(
                surface=cue.surface,
                reading=piece.reading,
                pos1=piece.pos1,
                pos2=piece.pos2,
                has_kanji=piece.has_kanji,
                punctuation=piece.punctuation,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return timed_tokens


def _shift_tokens(tokens: list[TimedToken], delta_ms: int) -> list[TimedToken]:
    return [
        TimedToken(
            surface=token.surface,
            reading=token.reading,
            pos1=token.pos1,
            pos2=token.pos2,
            has_kanji=token.has_kanji,
            punctuation=token.punctuation,
            start_ms=token.start_ms + delta_ms,
            end_ms=token.end_ms + delta_ms,
        )
        for token in tokens
    ]


def _stabilize_matched_cues(
    cues: list[AudioWordCue],
    line_end_ms: int,
) -> list[AudioWordCue]:
    stabilized: list[AudioWordCue] = []
    index = 0

    while index < len(cues):
        cue = cues[index]
        next_cue = cues[index + 1] if index + 1 < len(cues) else None

        if cue.end_ms - cue.start_ms < MIN_ALIGNMENT_CUE_MS:
            if next_cue and next_cue.start_ms <= cue.end_ms:
                cues[index + 1] = AudioWordCue(
                    surface=cue.surface + next_cue.surface,
                    start_ms=cue.start_ms,
                    end_ms=next_cue.end_ms,
                )
                index += 1
                continue

            previous_cue = stabilized[-1] if stabilized else None
            cue = AudioWordCue(
                surface=cue.surface,
                start_ms=cue.start_ms,
                end_ms=_fallback_cue_end_ms(cue.start_ms, previous_cue, next_cue, line_end_ms),
            )

        stabilized.append(cue)
        index += 1

    return stabilized


def _fallback_cue_end_ms(
    start_ms: int,
    previous_cue: Optional[AudioWordCue],
    next_cue: Optional[AudioWordCue],
    line_end_ms: int,
) -> int:
    default_duration_ms = max(80, min(260, max(1, line_end_ms - start_ms)))
    candidate_durations = [default_duration_ms]

    if previous_cue is not None and start_ms > previous_cue.end_ms:
        candidate_durations.append(start_ms - previous_cue.end_ms)
    if next_cue is not None and next_cue.start_ms > start_ms:
        candidate_durations.append(next_cue.start_ms - start_ms)

    duration_ms = min(duration for duration in candidate_durations if duration > 0)
    return min(line_end_ms, start_ms + max(1, duration_ms))


def _transcribe_line_cues(
    model: "WhisperModel",
    audio: np.ndarray,
    line: TimedLine,
) -> list[AudioWordCue]:
    clip_start_ms = max(0, line.start_ms - ALIGNMENT_PRE_ROLL_MS)
    clip_end_ms = min(len(audio) * 1000 // ALIGNMENT_SAMPLE_RATE, line.end_ms + ALIGNMENT_POST_ROLL_MS)
    start_index = max(0, round(clip_start_ms * ALIGNMENT_SAMPLE_RATE / 1000))
    end_index = min(len(audio), round(clip_end_ms * ALIGNMENT_SAMPLE_RATE / 1000))
    clip_audio = audio[start_index:end_index]
    if len(clip_audio) == 0:
        return []

    segments, _ = model.transcribe(
        clip_audio,
        language="ja",
        vad_filter=False,
        word_timestamps=True,
        beam_size=1,
        temperature=0,
        condition_on_previous_text=False,
        prefix=line.text,
    )

    cues: list[AudioWordCue] = []
    for segment in segments:
        for word in segment.words or []:
            surface = (word.word or "").strip()
            if not surface:
                continue
            start_ms = clip_start_ms + round(word.start * 1000)
            end_ms = clip_start_ms + round(word.end * 1000)
            cues.append(AudioWordCue(surface=surface, start_ms=start_ms, end_ms=end_ms))
    return cues


def _match_cues_to_line_text(
    text: str,
    cues: list[AudioWordCue],
    tokenizer: JapaneseTokenizer,
) -> list[AudioWordCue]:
    matched: list[AudioWordCue] = []
    cursor = 0

    for cue in cues:
        if cursor >= len(text):
            break

        leading_punctuation = ""
        while cursor < len(text) and _is_punctuation(text[cursor]):
            leading_punctuation += text[cursor]
            cursor += 1

        if cursor >= len(text):
            break

        fragment = _match_cue_surface_to_remaining_text(cue.surface, text[cursor:], tokenizer)
        if not fragment:
            continue

        if leading_punctuation:
            fragment = leading_punctuation + fragment

        matched.append(
            AudioWordCue(
                surface=fragment,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
            )
        )
        cursor += len(fragment) - len(leading_punctuation)

    if cursor < len(text):
        trailing = text[cursor:]
        if matched and all(_is_punctuation(char) for char in trailing):
            last = matched.pop()
            matched.append(
                AudioWordCue(
                    surface=last.surface + trailing,
                    start_ms=last.start_ms,
                    end_ms=last.end_ms,
                )
            )
            cursor = len(text)

    if cursor < len(text):
        raise RuntimeError(f"Unmatched lyric suffix during alignment: {text[cursor:]} in {text}")

    return matched


def _match_cue_surface_to_remaining_text(
    cue_surface: str,
    remaining_text: str,
    tokenizer: JapaneseTokenizer,
) -> str:
    cue_surface = cue_surface.strip()
    if not cue_surface or not remaining_text:
        return ""

    prefix_length = _longest_common_prefix(cue_surface, remaining_text)
    if prefix_length > 0:
        return remaining_text[:prefix_length]

    cue_reading = _reading_for_surface(cue_surface, tokenizer)
    best = ""
    max_length = min(len(remaining_text), max(len(cue_surface) + 2, 1))
    for length in range(1, max_length + 1):
        candidate = remaining_text[:length]
        candidate_reading = _reading_for_surface(candidate, tokenizer)
        if not candidate_reading:
            continue
        if cue_reading.startswith(candidate_reading) or candidate_reading.startswith(cue_reading):
            if len(candidate) > len(best):
                best = candidate
    return best


def _reading_for_surface(surface: str, tokenizer: JapaneseTokenizer) -> str:
    pieces = tokenizer.tokenize(surface)
    if not pieces:
        return _kata_to_hiragana(surface)
    return "".join(piece.reading for piece in pieces)


def _longest_common_prefix(left: str, right: str) -> int:
    prefix_length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        prefix_length += 1
    return prefix_length


def _token_piece_from_surface(surface: str, tokenizer: JapaneseTokenizer) -> TokenPiece:
    pieces = tokenizer.tokenize(surface)
    if not pieces:
        return TokenPiece(
            surface=surface,
            reading=_kata_to_hiragana(surface),
            has_kanji=bool(KANJI_RE.search(surface)),
            punctuation=_is_punctuation(surface),
        )

    if len(pieces) == 1 and pieces[0].surface == surface:
        return pieces[0]

    return TokenPiece(
        surface=surface,
        reading="".join(piece.reading for piece in pieces),
        pos1=pieces[0].pos1,
        pos2=pieces[-1].pos2,
        has_kanji=any(piece.has_kanji for piece in pieces),
        punctuation=all(piece.punctuation for piece in pieces),
    )


def _remap_ruby_groups(
    text: str,
    ruby_groups: list[RubyGroup],
    display_tokens: list[TimedToken],
) -> list[RubyGroup]:
    display_spans = _token_char_spans(display_tokens)
    remapped: list[RubyGroup] = []

    for group in ruby_groups:
        overlapping_indexes = [
            index
            for index, (start_char, end_char) in enumerate(display_spans)
            if end_char > group.char_start and start_char < group.char_end
        ]
        if not overlapping_indexes:
            continue

        token_start = overlapping_indexes[0]
        token_end = overlapping_indexes[-1] + 1
        remapped.append(
            RubyGroup(
                surface=text[group.char_start : group.char_end],
                reading=group.reading,
                token_start=token_start,
                token_end=token_end,
                char_start=group.char_start,
                char_end=group.char_end,
                start_ms=display_tokens[token_start].start_ms,
                end_ms=display_tokens[token_end - 1].end_ms,
            )
        )

    return remapped


def render_video(
    audio_path: Path,
    background_path: Path,
    ass_path: Path,
    output_path: Path,
    duration_ms: int,
    resolution: str,
    font_path: Optional[Path] = None,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found in PATH")

    width, height = RESOLUTION_MAP[resolution]
    bg_input = (
        ["-stream_loop", "-1", "-i", str(background_path)]
        if background_path.suffix.lower() in VIDEO_EXTENSIONS
        else ["-loop", "1", "-framerate", "30", "-i", str(background_path)]
    )
    subtitles_filter = _build_subtitles_filter(ass_path, font_path.parent if font_path else None)
    filter_complex = ";".join(
        [
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1[bg]",
            f"[bg]{subtitles_filter}[vout]",
        ]
    )

    command = [
        ffmpeg,
        "-y",
        *bg_input,
        "-i",
        str(audio_path),
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffmpeg export failed")
    return output_path


def render_local_jp_video(
    audio_path: Path,
    background_path: Path,
    output_dir: Path,
    output_stem: str,
    timing_lrc_path: Optional[Path] = None,
    lyrics_txt_path: Optional[Path] = None,
    txt_timing_bridge_lrc_path: Optional[Path] = None,
    resolution: str = "1080p",
) -> RenderResult:
    if resolution not in RESOLUTION_MAP:
        raise ValueError(f"Unsupported resolution: {resolution}")
    if timing_lrc_path is None and lyrics_txt_path is None:
        raise ValueError("Provide at least one of timing_lrc_path or lyrics_txt_path")
    if timing_lrc_path is not None and lyrics_txt_path is not None:
        raise ValueError("Provide timing_lrc_path for direct LRC render or lyrics_txt_path for TXT render, not both")

    output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = _probe_duration_ms(audio_path)
    tokenizer = JapaneseTokenizer()

    if timing_lrc_path is not None:
        raw_lrc = timing_lrc_path.read_text(encoding="utf-8")
        timed_lines = parse_lrc_timed_lines(raw_lrc, tokenizer=tokenizer)
        if not timed_lines:
            raise ValueError(f"No timed lyric lines found in {timing_lrc_path}")
        duration_ms = duration_ms or max(line.end_ms for line in timed_lines)
    else:
        raw_txt = lyrics_txt_path.read_text(encoding="utf-8")
        bridge_lines = None
        if txt_timing_bridge_lrc_path is not None:
            bridge_lines = parse_lrc_timed_lines(
                txt_timing_bridge_lrc_path.read_text(encoding="utf-8"),
                tokenizer=tokenizer,
            )
            if bridge_lines:
                duration_ms = duration_ms or max(line.end_ms for line in bridge_lines)
        if duration_ms is None:
            duration_ms = _estimate_txt_duration_ms(raw_txt)
        timed_lines = parse_txt_timed_lines(
            raw_txt,
            duration_ms=duration_ms,
            tokenizer=tokenizer,
            timing_bridge_lines=bridge_lines,
        )
        if not timed_lines:
            raise ValueError(f"No lyric lines found in {lyrics_txt_path}")

    timed_lines = _align_timed_lines_to_audio(
        timed_lines=timed_lines,
        audio_path=audio_path,
        tokenizer=tokenizer,
    )

    font_path = detect_font_path()
    fonts = load_font_set(font_path)
    font_family = _font_family_name(font_path)

    ass_path = output_dir / f"{output_stem}.ass"
    timeline_path = output_dir / f"{output_stem}.timeline.json"
    video_path = output_dir / f"{output_stem}.mp4"
    txt_path = write_txt_sidecar(timed_lines, output_dir / f"{output_stem}.txt")
    lrc_path = write_lrc_sidecar(timed_lines, output_dir / f"{output_stem}.lrc")

    write_ass_subtitles(
        timed_lines=timed_lines,
        output_path=ass_path,
        resolution=resolution,
        fonts=fonts,
        font_family=font_family,
    )
    write_timeline_debug(timed_lines, timeline_path)

    final_duration_ms = duration_ms or max(line.end_ms for line in timed_lines)
    render_video(
        audio_path=audio_path,
        background_path=background_path,
        ass_path=ass_path,
        output_path=video_path,
        duration_ms=final_duration_ms,
        resolution=resolution,
        font_path=font_path,
    )

    return RenderResult(
        video_path=video_path,
        ass_path=ass_path,
        timeline_path=timeline_path,
        txt_path=txt_path,
        lrc_path=lrc_path,
        duration_ms=final_duration_ms,
    )


def _parse_ruby_directives(raw_text: str) -> list[RubyDirective]:
    directives: list[RubyDirective] = []
    for raw_line in raw_text.splitlines():
        match = RUBY_DIRECTIVE_RE.match(raw_line.strip())
        if not match:
            continue
        directives.append(
            RubyDirective(
                index=int(match.group("index")),
                base_text=match.group("base").strip(),
                ruby_text=match.group("reading").strip(),
            )
        )
    return sorted(directives, key=lambda item: item.index)


def _resolve_ruby_directives_for_line_texts(
    line_texts: list[str],
    directives: list[RubyDirective],
) -> dict[int, list[ResolvedRubyDirective]]:
    if not directives or not line_texts:
        return {}

    compact_lines = [_compact_text_with_index_map(text) for text in line_texts]
    used_spans = {index: [] for index in range(len(line_texts))}
    resolved: dict[int, list[ResolvedRubyDirective]] = {}
    line_index = 0

    for directive in directives:
        normalized_base = "".join(char for char in directive.base_text if not char.isspace())
        if not normalized_base:
            continue

        for candidate_index in range(line_index, len(line_texts)):
            compact_text, index_map = compact_lines[candidate_index]
            if not compact_text:
                continue

            compact_match = _find_unused_match(compact_text, normalized_base, used_spans[candidate_index])
            if compact_match is None:
                continue

            compact_start, compact_end = compact_match
            used_spans[candidate_index].append((compact_start, compact_end))
            resolved.setdefault(candidate_index, []).append(
                ResolvedRubyDirective(
                    index=directive.index,
                    base_text=directive.base_text,
                    ruby_text=directive.ruby_text,
                    start_char=index_map[compact_start],
                    end_char=index_map[compact_end - 1] + 1,
                )
            )
            line_index = candidate_index
            break

    for items in resolved.values():
        items.sort(key=lambda item: (item.start_char, item.end_char, item.index))
    return resolved


def _compact_text_with_index_map(text: str) -> tuple[str, list[int]]:
    compact_chars: list[str] = []
    index_map: list[int] = []

    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        index_map.append(index)

    return "".join(compact_chars), index_map


def _find_unused_match(
    compact_text: str,
    needle: str,
    used_spans: list[tuple[int, int]],
) -> Optional[tuple[int, int]]:
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


def _ruby_groups_from_resolved_directives(
    text: str,
    tokens: list[TimedToken],
    resolved_directives: list[ResolvedRubyDirective],
) -> list[RubyGroup]:
    if not resolved_directives or not tokens:
        return []

    token_spans = _token_char_spans(tokens)
    groups: list[RubyGroup] = []
    for directive in resolved_directives:
        overlapping_indexes = [
            index
            for index, (start_char, end_char) in enumerate(token_spans)
            if end_char > directive.start_char and start_char < directive.end_char
        ]
        if not overlapping_indexes:
            continue
        token_start = overlapping_indexes[0]
        token_end = overlapping_indexes[-1] + 1
        groups.append(
            RubyGroup(
                surface=text[directive.start_char : directive.end_char],
                reading=directive.ruby_text,
                token_start=token_start,
                token_end=token_end,
                char_start=directive.start_char,
                char_end=directive.end_char,
                start_ms=tokens[token_start].start_ms,
                end_ms=tokens[token_end - 1].end_ms,
            )
        )
    return groups


def _token_char_spans(tokens: list[TimedToken]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start_char = cursor
        end_char = start_char + len(token.surface)
        spans.append((start_char, end_char))
        cursor = end_char
    return spans


def _retime_txt_lines_from_bridge(
    lyric_lines: list[str],
    timing_bridge_lines: list[TimedLine],
    tokenizer: JapaneseTokenizer,
) -> list[TimedLine]:
    if len(lyric_lines) != len(timing_bridge_lines):
        return []

    timed_lines: list[TimedLine] = []
    for text, bridge_line in zip(lyric_lines, timing_bridge_lines):
        token_pieces = tokenizer.tokenize(text)
        timed_tokens = _assign_weighted_token_timing(token_pieces, bridge_line.start_ms, bridge_line.end_ms)
        bridge_groups = _bridge_ruby_groups(text, timed_tokens, bridge_line.ruby_groups)
        timed_lines.append(
            TimedLine(
                start_ms=bridge_line.start_ms,
                end_ms=bridge_line.end_ms,
                text=text,
                tokens=timed_tokens,
                ruby_groups=bridge_groups or _build_ruby_groups(timed_tokens),
            )
        )
    return timed_lines


def _bridge_ruby_groups(
    text: str,
    tokens: list[TimedToken],
    bridge_groups: list[RubyGroup],
) -> list[RubyGroup]:
    directives = [
        RubyDirective(index=index, base_text=group.surface, ruby_text=group.reading)
        for index, group in enumerate(bridge_groups, start=1)
    ]
    resolved = _resolve_ruby_directives_for_line_texts([text], directives)
    return _ruby_groups_from_resolved_directives(text, tokens, resolved.get(0, []))


def _build_ruby_dialogue_lines(
    line: TimedLine,
    layout: LayoutMetrics,
    style: str,
    layer: int,
    start_ms: int,
    end_ms: int,
    ruby_y: int,
) -> list[str]:
    return [
        _format_dialogue(
            layer=layer,
            start_ms=start_ms,
            end_ms=end_ms,
            style=style,
            text=f"{{\\an8\\pos({_ruby_center_x(layout, group)},{ruby_y})\\q2}}{_escape_ass(group.reading)}",
        )
        for group in line.ruby_groups
    ]


def _build_karaoke_text(tokens: list[TimedToken], line_start_ms: int) -> str:
    parts: list[str] = []
    cursor = line_start_ms
    for token in tokens:
        gap_duration_ms = max(0, token.start_ms - cursor)
        if gap_duration_ms > 0:
            parts.append(r"{\k" + str(max(1, round(gap_duration_ms / 10))) + r"}")

        duration_cs = max(1, round((token.end_ms - token.start_ms) / 10))
        parts.append(r"{\kf" + str(duration_cs) + r"}" + _escape_ass(token.surface))
        cursor = token.end_ms
    return "".join(parts)


def _layout_line(
    draw: ImageDraw.ImageDraw,
    tokens: list[TimedToken],
    font: ImageFont.FreeTypeFont,
    canvas_width: int,
    line_index: int,
) -> LayoutMetrics:
    widths = [_measure_text_width(draw, token.surface, font) for token in tokens]
    total_width = sum(widths)
    inset = 116

    if line_index % 2 == 0:
        left = inset
        right = left + total_width
    else:
        right = canvas_width - inset
        left = right - total_width

    bounds: list[tuple[int, int]] = []
    char_bounds: list[tuple[int, int]] = []
    cursor = left
    for token, width in zip(tokens, widths):
        token_left = cursor
        token_right = token_left + width
        bounds.append((token_left, token_right))

        char_cursor = token_left
        for char_index in range(len(token.surface)):
            prefix = token.surface[: char_index + 1]
            char_right = token_left + _measure_text_width(draw, prefix, font)
            char_bounds.append((char_cursor, char_right))
            char_cursor = char_right

        cursor = token_right
    return LayoutMetrics(
        left=left,
        right=right,
        width=total_width,
        token_bounds=bounds,
        char_bounds=char_bounds,
    )


def _line_anchor(line_index: int, width: int, height: int) -> tuple[int, int, int]:
    left_inset = 116
    right_inset = width - 116
    top_y = height - 292
    bottom_y = height - 162
    if line_index % 2 == 0:
        return 7, left_inset, top_y
    return 9, right_inset, bottom_y


def _ruby_y(line_y: int) -> int:
    return line_y - 54


def _ruby_center_x(layout: LayoutMetrics, ruby_group: RubyGroup) -> int:
    if layout.char_bounds and ruby_group.char_end > ruby_group.char_start:
        start_left = layout.char_bounds[ruby_group.char_start][0]
        end_right = layout.char_bounds[ruby_group.char_end - 1][1]
        return (start_left + end_right) // 2

    start_left = layout.token_bounds[ruby_group.token_start][0]
    end_right = layout.token_bounds[ruby_group.token_end - 1][1]
    return (start_left + end_right) // 2


def _build_ruby_groups(tokens: list[TimedToken]) -> list[RubyGroup]:
    lyric_text = "".join(token.surface for token in tokens)
    tokenizer = JapaneseTokenizer()
    lyric_tokens = tokenizer.tokenize(lyric_text)
    display_spans = _token_char_spans(tokens)

    groups: list[RubyGroup] = []
    cursor = 0
    for piece in lyric_tokens:
        piece_start = cursor
        piece_end = piece_start + len(piece.surface)
        cursor = piece_end

        if not piece.has_kanji or piece.punctuation:
            continue

        for token_char_start, token_char_end, reading in _token_ruby_segments(piece.surface, piece.reading):
            full_char_start = piece_start + token_char_start
            full_char_end = piece_start + token_char_end
            overlapping_indexes = [
                index
                for index, (start_char, end_char) in enumerate(display_spans)
                if end_char > full_char_start and start_char < full_char_end
            ]
            if not overlapping_indexes or not reading:
                continue

            token_start = overlapping_indexes[0]
            token_end = overlapping_indexes[-1] + 1
            groups.append(
                RubyGroup(
                    surface=lyric_text[full_char_start:full_char_end],
                    reading=reading,
                    token_start=token_start,
                    token_end=token_end,
                    char_start=full_char_start,
                    char_end=full_char_end,
                    start_ms=tokens[token_start].start_ms,
                    end_ms=tokens[token_end - 1].end_ms,
                )
            )

    return groups


def _token_ruby_segments(surface: str, reading: str) -> list[tuple[int, int, str]]:
    normalized_reading = _kata_to_hiragana(reading or surface)
    if not surface or not normalized_reading or not KANJI_RE.search(surface):
        return []

    parts: list[tuple[bool, str, int, int]] = []
    start = 0
    while start < len(surface):
        is_kanji = bool(KANJI_RE.fullmatch(surface[start]))
        end = start + 1
        while end < len(surface) and bool(KANJI_RE.fullmatch(surface[end])) == is_kanji:
            end += 1
        parts.append((is_kanji, surface[start:end], start, end))
        start = end

    reading_cursor = 0
    pending_run: Optional[tuple[int, int, int]] = None
    segments: list[tuple[int, int, str]] = []

    for is_kanji, part_text, part_start, part_end in parts:
        if is_kanji:
            pending_run = (part_start, part_end, reading_cursor)
            continue

        anchor = _normalize_ruby_anchor(part_text)
        if not anchor:
            continue

        match_index = normalized_reading.find(anchor, reading_cursor)
        if pending_run is not None:
            run_start, run_end, reading_start = pending_run
            ruby_reading = normalized_reading[reading_start:match_index] if match_index >= reading_start else ""
            if ruby_reading:
                segments.append((run_start, run_end, ruby_reading))
            pending_run = None

        if match_index == -1:
            reading_cursor = min(len(normalized_reading), reading_cursor + len(anchor))
        else:
            reading_cursor = match_index + len(anchor)

    if pending_run is not None:
        run_start, run_end, reading_start = pending_run
        ruby_reading = normalized_reading[reading_start:]
        if ruby_reading:
            segments.append((run_start, run_end, ruby_reading))

    return segments


def _normalize_ruby_anchor(text: str) -> str:
    if _is_punctuation(text):
        return ""
    return _kata_to_hiragana(text)


def _should_extend_ruby_group(head: TimedToken, last: TimedToken, next_token: TimedToken) -> bool:
    if next_token.punctuation or next_token.has_kanji:
        return False
    if next_token.surface in {"て", "で"} and (
        head.pos1 in {"動詞", "形容詞", "形状詞"} or last.pos1 in {"動詞", "助動詞", "接尾辞"}
    ):
        return True
    if next_token.surface in PARTICLE_SURFACES or next_token.pos1 == "助詞":
        return False
    if head.pos1 in {"動詞", "形容詞", "形状詞"} and next_token.pos1 in {"助動詞", "接尾辞"}:
        return True
    if last.pos1 in {"助動詞", "接尾辞"} and next_token.pos1 in {"動詞", "助動詞"}:
        return True
    if KANJI_ONLY_RE.fullmatch(head.surface or "") and len(head.surface) == 1 and next_token.pos1 in {
        "動詞",
        "形容詞",
        "形状詞",
        "助動詞",
    }:
        return True
    return False


def _assign_weighted_token_timing(tokens: list[TokenPiece], start_ms: int, end_ms: int) -> list[TimedToken]:
    if not tokens:
        return []

    weights = [_mora_weight(token) for token in tokens]
    total_weight = max(sum(weights), 1)
    total_duration = max(end_ms - start_ms, len(tokens) * 120)

    timed_tokens: list[TimedToken] = []
    cursor = start_ms
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            token_end = end_ms
        else:
            token_duration = max(90, round(total_duration * weights[index] / total_weight))
            token_end = min(end_ms, cursor + token_duration)
        if token_end <= cursor:
            token_end = min(end_ms, cursor + 90)
        timed_tokens.append(
            TimedToken(
                surface=token.surface,
                reading=token.reading,
                pos1=token.pos1,
                pos2=token.pos2,
                has_kanji=token.has_kanji,
                punctuation=token.punctuation,
                start_ms=cursor,
                end_ms=token_end,
            )
        )
        cursor = token_end

    last_token = timed_tokens[-1]
    timed_tokens[-1] = TimedToken(
        surface=last_token.surface,
        reading=last_token.reading,
        pos1=last_token.pos1,
        pos2=last_token.pos2,
        has_kanji=last_token.has_kanji,
        punctuation=last_token.punctuation,
        start_ms=last_token.start_ms,
        end_ms=end_ms,
    )
    return timed_tokens


def _mora_weight(token: TokenPiece) -> int:
    if token.punctuation:
        return 1
    reading = token.reading or token.surface
    normalized = "".join(char for char in reading if char not in SMALL_KANA and char != "ー")
    return max(1, len(normalized) or len(reading) or len(token.surface))


def _reading_override(surface: str, base_reading: str, next_surface: str) -> str:
    if surface in READING_OVERRIDES:
        if surface != "私" or next_surface in PARTICLE_SURFACES or next_surface == "":
            return READING_OVERRIDES[surface]
    return base_reading


def _can_merge_numeric_tokens(previous: TokenPiece, current: TokenPiece) -> bool:
    return (
        previous.pos1 == "名詞"
        and current.pos1 == "名詞"
        and previous.pos2 == "数詞"
        and current.pos2 == "数詞"
        and KANJI_ONLY_RE.fullmatch(previous.surface or "")
        and KANJI_ONLY_RE.fullmatch(current.surface or "")
    )


def _font_family_name(font_path: Optional[Path]) -> str:
    if font_path and "NotoSansCJK" in font_path.name:
        return "Noto Sans CJK JP"
    if font_path:
        return font_path.stem
    return "Arial"


def _parse_timestamp(match: re.Match[str]) -> int:
    fraction = match.group(3) or "0"
    millis = int(fraction.ljust(3, "0")[:3])
    return int(match.group(1)) * 60_000 + int(match.group(2)) * 1_000 + millis


def _format_dialogue(layer: int, start_ms: int, end_ms: int, style: str, text: str) -> str:
    return f"Dialogue: {layer},{_format_ass_timestamp(start_ms)},{_format_ass_timestamp(end_ms)},{style},,0,0,0,,{text}"


def _format_ass_timestamp(milliseconds: int) -> str:
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1_000
    centiseconds = (milliseconds % 1_000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _format_lrc_timestamp(milliseconds: int) -> str:
    minutes = milliseconds // 60_000
    seconds = (milliseconds % 60_000) // 1_000
    centiseconds = (milliseconds % 1_000) // 10
    return f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]"


def _measure_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _build_subtitles_filter(ass_path: Path, fonts_dir: Optional[Path]) -> str:
    subtitles = f"subtitles=filename='{_escape_ffmpeg_filter_value(ass_path.resolve().as_posix())}'"
    if fonts_dir:
        subtitles += f":fontsdir='{_escape_ffmpeg_filter_value(fonts_dir.resolve().as_posix())}'"
    return subtitles


def _escape_ffmpeg_filter_value(value: str) -> str:
    escaped = value.replace("\\", r"\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace("[", r"\[").replace("]", r"\]")
    escaped = escaped.replace(",", r"\,").replace(";", r"\;")
    return escaped


def _probe_duration_ms(audio_path: Path) -> Optional[int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None

    try:
        return math.ceil(float(json.loads(result.stdout)["format"]["duration"]) * 1000)
    except Exception:
        return None


def _extract_lyric_text_lines(raw_text: str) -> list[str]:
    lyric_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        stripped = raw_line.lstrip("\ufeff").strip()
        if not stripped or stripped.startswith("@Ruby"):
            continue
        cleaned = TIMESTAMP_RE.sub("", stripped).strip()
        if cleaned:
            lyric_lines.append(cleaned)
    return lyric_lines


def _derive_txt_timing_window(duration_ms: int, line_count: int) -> tuple[int, int]:
    minimum_content_ms = max(line_count, 1) * TXT_MIN_LINE_DURATION_MS
    available_padding_ms = max(duration_ms - minimum_content_ms, 0)
    desired_lead_in_ms = _clamp(round(duration_ms * TXT_LEAD_IN_RATIO), TXT_MIN_LEAD_IN_MS, TXT_MAX_LEAD_IN_MS)
    desired_lead_out_ms = _clamp(round(duration_ms * TXT_LEAD_OUT_RATIO), TXT_MIN_LEAD_OUT_MS, TXT_MAX_LEAD_OUT_MS)
    desired_padding_ms = desired_lead_in_ms + desired_lead_out_ms
    applied_padding_ms = min(desired_padding_ms, available_padding_ms)

    if applied_padding_ms <= 0 or desired_padding_ms <= 0:
        return 0, duration_ms

    lead_in_ms = round(applied_padding_ms * desired_lead_in_ms / desired_padding_ms)
    lead_out_ms = applied_padding_ms - lead_in_ms
    content_end_ms = max(duration_ms - lead_out_ms, lead_in_ms + minimum_content_ms)
    return lead_in_ms, content_end_ms


def _estimate_txt_duration_ms(raw_text: str) -> int:
    lyric_lines = _extract_lyric_text_lines(raw_text)
    if not lyric_lines:
        return TXT_TARGET_LINE_DURATION_MS

    estimated_content_ms = max(
        len(lyric_lines) * TXT_TARGET_LINE_DURATION_MS,
        sum(max(len(line), 1) * 180 for line in lyric_lines),
    )
    desired_lead_in_ms = _clamp(
        round(estimated_content_ms * TXT_LEAD_IN_RATIO),
        TXT_MIN_LEAD_IN_MS,
        TXT_MAX_LEAD_IN_MS,
    )
    desired_lead_out_ms = _clamp(
        round(estimated_content_ms * TXT_LEAD_OUT_RATIO),
        TXT_MIN_LEAD_OUT_MS,
        TXT_MAX_LEAD_OUT_MS,
    )
    return estimated_content_ms + desired_lead_in_ms + desired_lead_out_ms


def _line_weight(tokens: list[TokenPiece]) -> int:
    return max(1, sum(_mora_weight(token) for token in tokens))


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _kata_to_hiragana(text: str) -> str:
    converted: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            converted.append(chr(codepoint - 0x60))
        else:
            converted.append(char)
    return "".join(converted)


def _is_punctuation(text: str) -> bool:
    return bool(
        text
        and not any(
            char.isalnum() or KANJI_RE.search(char) or "ぁ" <= char <= "ゖ" or "ァ" <= char <= "ヺ"
            for char in text
        )
    )
