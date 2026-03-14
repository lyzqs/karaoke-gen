from karaoke_gen.lyrics_transcriber.ruby import RubyAnnotation, parse_ruby_annotations_from_lrc_text, resolve_ruby_annotations_for_segments
from karaoke_gen.lyrics_transcriber.types import LyricsSegment, Word



def _make_segment(segment_id: str, text: str) -> LyricsSegment:
    return LyricsSegment(
        id=segment_id,
        text=text,
        words=[
            Word(
                id=f"{segment_id}-word-1",
                text=text,
                start_time=0.0,
                end_time=1.0,
                confidence=1.0,
            )
        ],
        start_time=0.0,
        end_time=1.0,
    )



def test_parse_ruby_annotations_from_lrc_text_sorts_by_index():
    lrc_text = """
[00:01.00]君と一緒ならば
@Ruby2=一緒,いっしょ
@Ruby1=君,きみ
@Ruby4=パノラマ,panorama
@Ruby3=広,ひろ
""".strip()

    annotations = parse_ruby_annotations_from_lrc_text(lrc_text)

    assert annotations == [
        RubyAnnotation(index=1, base_text="君", ruby_text="きみ"),
        RubyAnnotation(index=2, base_text="一緒", ruby_text="いっしょ"),
        RubyAnnotation(index=3, base_text="広", ruby_text="ひろ"),
        RubyAnnotation(index=4, base_text="パノラマ", ruby_text="panorama"),
    ]



def test_resolve_ruby_annotations_for_segments_matches_in_output_order():
    segments = [
        _make_segment("seg-1", "君と一緒ならば"),
        _make_segment("seg-2", "広がるパノラマ"),
    ]
    annotations = [
        RubyAnnotation(index=1, base_text="君", ruby_text="きみ"),
        RubyAnnotation(index=2, base_text="一緒", ruby_text="いっしょ"),
        RubyAnnotation(index=3, base_text="広", ruby_text="ひろ"),
        RubyAnnotation(index=4, base_text="パノラマ", ruby_text="panorama"),
    ]

    resolved = resolve_ruby_annotations_for_segments(segments, annotations)

    assert [annotation.base_text for annotation in resolved[0]] == ["君", "一緒"]
    assert [annotation.ruby_text for annotation in resolved[0]] == ["きみ", "いっしょ"]
    assert resolved[0][0].start_char == segments[0].text.index("君")
    assert resolved[0][1].start_char == segments[0].text.index("一緒")

    assert [annotation.base_text for annotation in resolved[1]] == ["広", "パノラマ"]
    assert [annotation.ruby_text for annotation in resolved[1]] == ["ひろ", "panorama"]
    assert resolved[1][0].start_char == segments[1].text.index("広")
    assert resolved[1][1].start_char == segments[1].text.index("パノラマ")
