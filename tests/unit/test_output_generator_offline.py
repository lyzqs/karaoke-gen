import pytest

from karaoke_gen.lyrics_transcriber.core.config import OutputConfig
from karaoke_gen.lyrics_transcriber.output.countdown_processor import CountdownProcessor
from karaoke_gen.lyrics_transcriber.output.generator import OutputGenerator
from karaoke_gen.lyrics_transcriber.types import (
    CorrectionResult,
    LyricsData,
    LyricsMetadata,
    LyricsSegment,
    Word,
)


def _make_segment(segment_id: str, text: str, start_time: float, end_time: float) -> LyricsSegment:
    words = text.split()
    if not words:
        words = [text]

    step = (end_time - start_time) / len(words) if words else 0.0
    segment_words = []
    for index, word_text in enumerate(words):
        word_start = start_time + (step * index)
        word_end = end_time if index == len(words) - 1 else start_time + (step * (index + 1))
        segment_words.append(
            Word(
                id=f"{segment_id}-word-{index}",
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                confidence=1.0,
            )
        )

    return LyricsSegment(
        id=segment_id,
        text=text,
        words=segment_words,
        start_time=start_time,
        end_time=end_time,
    )


def test_prepare_correction_for_output_prefers_file_reference_and_strips_countdown(tmp_path):
    canonical_segments = [
        _make_segment("ref-1", "canon one", 0.0, 0.0),
        _make_segment("ref-2", "canon two", 0.0, 0.0),
    ]
    correction_result = CorrectionResult(
        original_segments=[],
        corrected_segments=[
            _make_segment("countdown", CountdownProcessor.COUNTDOWN_TEXT, 0.1, 2.9),
            _make_segment("seg-1", "please one", 3.0, 4.0),
            _make_segment("seg-2", "thank two", 4.0, 5.0),
        ],
        corrections=[],
        corrections_made=0,
        confidence=1.0,
        reference_lyrics={
            "file": LyricsData(
                source="file",
                segments=canonical_segments,
                metadata=LyricsMetadata(
                    source="file",
                    track_name="Track",
                    artist_names="Artist",
                    is_synced=False,
                    lyrics_provider="file",
                    lyrics_provider_id="lyrics.txt",
                ),
            )
        },
        anchor_sequences=[],
        gap_sequences=[],
        resized_segments=[],
        metadata={},
        correction_steps=[],
        word_id_map={},
        segment_id_map={},
    )

    generator = OutputGenerator(
        config=OutputConfig(
            output_styles_json="",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            render_video=False,
            generate_cdg=False,
            prefer_reference_lyrics_source="file",
            strip_countdown_text=True,
        )
    )

    prepared = generator._prepare_correction_for_output(correction_result)

    assert [segment.text for segment in prepared.corrected_segments] == ["canon one", "canon two"]
    assert [word.text for word in prepared.corrected_segments[0].words] == ["canon", "one"]
    assert [word.text for word in prepared.corrected_segments[1].words] == ["canon", "two"]
    assert prepared.corrected_segments[0].start_time == pytest.approx(3.0)
    assert prepared.corrected_segments[1].end_time == pytest.approx(5.0)


def test_prepare_correction_for_output_drops_extra_timed_tail_segments(tmp_path):
    canonical_segments = [
        _make_segment("ref-1", "canon one", 0.0, 0.0),
        _make_segment("ref-2", "canon two", 0.0, 0.0),
    ]
    correction_result = CorrectionResult(
        original_segments=[],
        corrected_segments=[
            _make_segment("countdown", CountdownProcessor.COUNTDOWN_TEXT, 0.1, 2.9),
            _make_segment("seg-1", "canon one", 3.0, 4.0),
            _make_segment("seg-2", "canon two", 4.0, 5.0),
            _make_segment("seg-3", "please thank", 5.0, 6.0),
        ],
        corrections=[],
        corrections_made=0,
        confidence=1.0,
        reference_lyrics={
            "file": LyricsData(
                source="file",
                segments=canonical_segments,
                metadata=LyricsMetadata(
                    source="file",
                    track_name="Track",
                    artist_names="Artist",
                    is_synced=False,
                    lyrics_provider="file",
                    lyrics_provider_id="lyrics.txt",
                ),
            )
        },
        anchor_sequences=[],
        gap_sequences=[],
        resized_segments=[],
        metadata={},
        correction_steps=[],
        word_id_map={},
        segment_id_map={},
    )

    generator = OutputGenerator(
        config=OutputConfig(
            output_styles_json="",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            render_video=False,
            generate_cdg=False,
            prefer_reference_lyrics_source="file",
            strip_countdown_text=True,
        )
    )

    prepared = generator._prepare_correction_for_output(correction_result)

    assert [segment.id for segment in prepared.corrected_segments] == ["ref-1", "ref-2"]
    assert [segment.text for segment in prepared.corrected_segments] == ["canon one", "canon two"]
    assert prepared.corrected_segments[0].start_time == pytest.approx(3.0)
    assert prepared.corrected_segments[1].end_time == pytest.approx(5.0)
