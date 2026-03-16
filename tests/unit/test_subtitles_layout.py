import re

from karaoke_gen.style_loader import get_default_style_params
from karaoke_gen.lyrics_transcriber.output.subtitles import SubtitlesGenerator
from karaoke_gen.lyrics_transcriber.ruby import RubyAnnotation
from karaoke_gen.lyrics_transcriber.types import LyricsSegment, Word


def _make_segment(segment_id: str, text: str, start_time: float, end_time: float) -> LyricsSegment:
    return LyricsSegment(
        id=segment_id,
        text=text,
        words=[
            Word(
                id=f"{segment_id}-word-1",
                text=text,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,
            )
        ],
        start_time=start_time,
        end_time=end_time,
    )


def _make_generator() -> SubtitlesGenerator:
    styles = get_default_style_params()
    styles["karaoke"]["lead_in_enabled"] = False
    return SubtitlesGenerator(
        output_dir="/tmp",
        video_resolution=(640, 360),
        font_size=40,
        line_height=50,
        styles=styles,
    )


def _extract_pos(text: str) -> tuple[int, int]:
    match = re.search(r"\\pos\((\d+),(\d+)\)", text)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def test_create_screens_hides_section_marker_text_but_keeps_lyric_screens():
    generator = _make_generator()
    segments = [
        _make_segment("seg-1", "君と一緒ならば", 12.0, 13.0),
        _make_segment("seg-2", "広がるパノラマ", 28.0, 29.0),
    ]

    screens = generator._create_screens(segments, song_duration=40.0)

    assert len(screens) == 2
    assert all(not hasattr(screen, "section_type") for screen in screens)

    ass = generator._create_styled_subtitles(screens, (640, 360), 40)
    all_text = "\n".join(event.Text for event in ass.events)

    assert "INTRO" not in all_text
    assert "OUTRO" not in all_text
    assert "INSTRUMENTAL" not in all_text


def test_styled_subtitles_use_lower_third_left_right_layout_and_ruby_tracks_line_anchor():
    generator = _make_generator()
    segments = [
        _make_segment("seg-1", "君と一緒ならば", 0.0, 1.0),
        _make_segment("seg-2", "君と一緒ならば", 1.2, 2.2),
    ]
    ruby_annotations = [
        RubyAnnotation(index=1, base_text="一緒", ruby_text="いっしょ"),
        RubyAnnotation(index=2, base_text="一緒", ruby_text="いっしょ"),
    ]

    screens = generator._create_screens(segments, song_duration=4.0, ruby_annotations=ruby_annotations)
    ass = generator._create_styled_subtitles(screens, (640, 360), 40)

    main_events = [event for event in ass.events if event.Layer == 0]
    ruby_events = [event for event in ass.events if event.Layer == 1]

    assert len(main_events) == 2
    assert len(ruby_events) == 2
    assert main_events[0].Text.startswith(r"{\an7}{\pos(")
    assert main_events[1].Text.startswith(r"{\an9}{\pos(")

    first_main_x, first_main_y = _extract_pos(main_events[0].Text)
    second_main_x, second_main_y = _extract_pos(main_events[1].Text)
    first_ruby_x, _ = _extract_pos(ruby_events[0].Text)
    second_ruby_x, _ = _extract_pos(ruby_events[1].Text)

    assert first_main_x < second_main_x
    assert first_main_y < second_main_y
    assert first_main_y >= 240
    assert second_main_y >= 280
    assert second_ruby_x > first_ruby_x
