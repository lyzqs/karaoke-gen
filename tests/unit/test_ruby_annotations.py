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


def test_resolve_ruby_annotations_for_segments_handles_reverse_visual_order_within_a_segment():
    segments = [
        _make_segment("seg-1", "君 を 見 つめた"),
        _make_segment("seg-2", "あ もう 気 にしてる 大 丈夫 かな"),
    ]
    annotations = [
        RubyAnnotation(index=1, base_text="見", ruby_text="み"),
        RubyAnnotation(index=2, base_text="君", ruby_text="きみ"),
        RubyAnnotation(index=3, base_text="気", ruby_text="き"),
        RubyAnnotation(index=4, base_text="大丈夫", ruby_text="だいじょうぶ"),
    ]

    resolved = resolve_ruby_annotations_for_segments(segments, annotations)
    resolved_by_index = {annotation.index: annotation for items in resolved.values() for annotation in items}

    assert [annotation.index for annotation in resolved[0]] == [2, 1]
    assert segments[0].text[resolved_by_index[2].start_char : resolved_by_index[2].end_char].replace(" ", "") == "君"
    assert segments[0].text[resolved_by_index[1].start_char : resolved_by_index[1].end_char].replace(" ", "") == "見"
    assert segments[1].text[resolved_by_index[4].start_char : resolved_by_index[4].end_char].replace(" ", "") == "大丈夫"



def test_resolve_ruby_annotations_for_segments_matches_split_compounds_without_poisoning_later_annotations():
    segments = [
        _make_segment("seg-1", "朝 日 みたい に 伝 えたい 君 に 今"),
        _make_segment("seg-2", "君 と 一 緒 ならば"),
        _make_segment("seg-3", "私 の こと を 全 然"),
        _make_segment("seg-4", "高 鳴 る 未 来 を 抜 いて"),
        _make_segment("seg-5", "この 瞬 間 いつまでも 守 りたい"),
    ]
    annotations = [
        RubyAnnotation(index=12, base_text="叫", ruby_text="さけ"),
        RubyAnnotation(index=13, base_text="伝", ruby_text="つた"),
        RubyAnnotation(index=14, base_text="今", ruby_text="いま"),
        RubyAnnotation(index=22, base_text="一緒", ruby_text="いっしょ"),
        RubyAnnotation(index=26, base_text="全然", ruby_text="ぜんぜん"),
        RubyAnnotation(index=40, base_text="未来", ruby_text="みらい"),
        RubyAnnotation(index=52, base_text="守", ruby_text="まも"),
        RubyAnnotation(index=53, base_text="瞬間", ruby_text="しゅんかん"),
    ]

    resolved = resolve_ruby_annotations_for_segments(segments, annotations)
    resolved_by_index = {annotation.index: (segment_index, annotation) for segment_index, items in resolved.items() for annotation in items}

    assert 12 not in resolved_by_index
    assert resolved_by_index[13][0] == 0
    assert resolved_by_index[14][0] == 0
    assert resolved_by_index[22][0] == 1
    assert resolved_by_index[26][0] == 2
    assert resolved_by_index[40][0] == 3
    assert resolved_by_index[53][0] == 4
    assert resolved_by_index[52][0] == 4
    assert [annotation.index for annotation in resolved[4]] == [53, 52]

    expected_matches = {
        13: "伝",
        14: "今",
        22: "一緒",
        26: "全然",
        40: "未来",
        52: "守",
        53: "瞬間",
    }

    for index, expected_base_text in expected_matches.items():
        segment_index, annotation = resolved_by_index[index]
        matched_text = segments[segment_index].text[annotation.start_char : annotation.end_char].replace(" ", "")
        assert matched_text == expected_base_text
