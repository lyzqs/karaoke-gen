from __future__ import annotations

from PIL import Image, ImageDraw
import pytest

import karaoke_gen.local_jp_renderer as local_jp_renderer
from karaoke_gen.local_jp_renderer import (
    build_ass_document,
    detect_font_path,
    load_font_set,
    parse_lrc_timed_lines,
    parse_txt_timed_lines,
    render_local_jp_video,
)


def test_parse_lrc_timed_lines_preserves_timestamps_and_resolves_explicit_ruby_directives():
    raw_lrc = "\n".join(
        [
            "[00:10.00]君を見つめた",
            "[00:12.50]好きな気持ちで",
            "@Ruby2=見,み",
            "@Ruby1=君,きみ",
        ]
    )

    timed_lines = parse_lrc_timed_lines(raw_lrc)

    assert [line.text for line in timed_lines] == ["君を見つめた", "好きな気持ちで"]
    assert timed_lines[0].start_ms == 10_000
    assert timed_lines[0].end_ms == 12_500
    assert [
        (group.surface, group.reading, group.char_start, group.char_end)
        for group in timed_lines[0].ruby_groups
    ] == [
        ("君", "きみ", 0, 1),
        ("見", "み", 2, 3),
    ]


def test_parse_txt_timed_lines_uses_bridge_timings_and_inherits_bridge_ruby_groups():
    bridge_lines = parse_lrc_timed_lines(
        "\n".join(
            [
                "[00:03.00]君を見つめた",
                "[00:05.50]好きな気持ちで",
                "@Ruby1=君,きみ",
                "@Ruby2=気持,きも",
            ]
        )
    )

    timed_lines = parse_txt_timed_lines(
        "君を見つめた\n好きな気持ちで\n",
        duration_ms=20_000,
        timing_bridge_lines=bridge_lines,
    )

    assert [line.text for line in timed_lines] == ["君を見つめた", "好きな気持ちで"]
    assert timed_lines[0].start_ms == 3_000
    assert timed_lines[0].end_ms == 5_500
    assert timed_lines[1].start_ms == 5_500
    assert timed_lines[1].end_ms > timed_lines[1].start_ms
    assert ("君", "きみ") in [(group.surface, group.reading) for group in timed_lines[0].ruby_groups]
    assert ("気持", "きも") in [(group.surface, group.reading) for group in timed_lines[1].ruby_groups]


def test_ass_document_uses_current_and_upcoming_lines_not_previous():
    font_path = detect_font_path()
    if font_path is None:
        pytest.skip("No CJK font available")

    timed_lines = parse_lrc_timed_lines(
        "\n".join(
            [
                "[00:00.00]君を見つめた",
                "[00:02.00]好きな気持ちで",
                "[00:04.00]抱きしめたいの",
            ]
        )
    )
    fonts = load_font_set(font_path)
    ass = build_ass_document(
        timed_lines=timed_lines,
        resolution="1080p",
        fonts=fonts,
        font_family="Noto Sans CJK JP",
    )

    assert (
        "Style: CurrentLine,Noto Sans CJK JP,80,"
        f"{local_jp_renderer.CURRENT_ACTIVE_FILL_ASS},"
        f"{local_jp_renderer.CURRENT_INACTIVE_FILL_ASS}"
    ) in ass
    assert "Style: UpcomingLine,Noto Sans CJK JP,80" in ass
    assert r"{\an7\pos(116,788)\q2}" in ass
    assert r"{\an9\pos(1804,918)\q2}" in ass
    assert "Dialogue: 0,0:00:02.00,0:00:04.00,CurrentLine" in ass
    assert "Dialogue: 1,0:00:02.00,0:00:04.00,UpcomingLine" in ass
    assert (
        r"{\1c"
        + local_jp_renderer.CURRENT_ACTIVE_FILL_ASS
        + r"\2c"
        + local_jp_renderer.CURRENT_INACTIVE_FILL_ASS
        + r"\3c"
        + local_jp_renderer.TEXT_OUTLINE_ASS
        + r"}"
    ) in ass

    second_window = ass.split("Dialogue: 0,0:00:02.00,0:00:04.00,CurrentLine", 1)[1]
    second_window = second_window.split("Dialogue: 0,0:00:04.00", 1)[0]
    assert "好き" in second_window
    assert "気持ち" in second_window
    assert "抱きしめたいの" in second_window
    assert "君を見つめた" not in second_window


def test_ruby_anchor_uses_exact_char_span_instead_of_full_token_bounds():
    font_path = detect_font_path()
    if font_path is None:
        pytest.skip("No CJK font available")

    timed_line = parse_lrc_timed_lines(
        "\n".join(
            [
                "[00:00.00]君を見つめた",
                "[00:02.00]次の行",
                "@Ruby1=見,み",
            ]
        )
    )[0]
    ruby_group = next(group for group in timed_line.ruby_groups if group.surface == "見")
    fonts = load_font_set(font_path)
    draw = ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    layout = local_jp_renderer._layout_line(
        draw=draw,
        tokens=timed_line.tokens,
        font=fonts.main,
        canvas_width=1920,
        line_index=0,
    )

    ruby_center_x = local_jp_renderer._ruby_center_x(layout, ruby_group)
    char_center_x = (
        layout.char_bounds[ruby_group.char_start][0] + layout.char_bounds[ruby_group.char_end - 1][1]
    ) // 2
    token_center_x = (
        layout.token_bounds[ruby_group.token_start][0] + layout.token_bounds[ruby_group.token_end - 1][1]
    ) // 2

    assert ruby_center_x == char_center_x
    assert ruby_center_x < token_center_x


def test_auto_ruby_groups_only_cover_mixed_token_kanji_chars():
    timed_line = parse_lrc_timed_lines(
        "\n".join(
            [
                "[00:00.00]見つめたい",
                "[00:02.00]心を語りたい",
            ]
        )
    )[0]

    mixed_group = next(group for group in timed_line.ruby_groups if group.surface == "見")

    assert mixed_group.reading == "み"
    assert mixed_group.char_start == 0
    assert mixed_group.char_end == 1


def test_match_cues_to_line_text_trims_trailing_hallucination_but_keeps_exact_lyric_surface():
    tokenizer = local_jp_renderer.JapaneseTokenizer()
    cues = [
        local_jp_renderer.AudioWordCue(surface="広", start_ms=0, end_ms=180),
        local_jp_renderer.AudioWordCue(surface="が", start_ms=180, end_ms=360),
        local_jp_renderer.AudioWordCue(surface="る", start_ms=360, end_ms=520),
        local_jp_renderer.AudioWordCue(surface="パ", start_ms=520, end_ms=700),
        local_jp_renderer.AudioWordCue(surface="ノ", start_ms=700, end_ms=860),
        local_jp_renderer.AudioWordCue(surface="ラ", start_ms=860, end_ms=1_020),
        local_jp_renderer.AudioWordCue(surface="マ", start_ms=1_020, end_ms=1_180),
        local_jp_renderer.AudioWordCue(surface="ー", start_ms=1_180, end_ms=1_360),
    ]

    matched = local_jp_renderer._match_cues_to_line_text("広がるパノラマ", cues, tokenizer)

    assert "".join(cue.surface for cue in matched) == "広がるパノラマ"
    assert [cue.surface for cue in matched] == ["広", "が", "る", "パ", "ノ", "ラ", "マ"]


def test_build_karaoke_text_inserts_real_timing_gaps_between_aligned_tokens():
    tokens = [
        local_jp_renderer.TimedToken(surface="君", reading="きみ", start_ms=220, end_ms=520),
        local_jp_renderer.TimedToken(surface="を", reading="を", start_ms=600, end_ms=820),
    ]

    karaoke_text = local_jp_renderer._build_karaoke_text(tokens, line_start_ms=100)

    assert karaoke_text.startswith(r"{\k12}{\kf30}君")
    assert r"{\k8}{\kf22}を" in karaoke_text


def test_stabilize_matched_cues_merges_zero_duration_prefix_into_following_cue():
    stabilized = local_jp_renderer._stabilize_matched_cues(
        [
            local_jp_renderer.AudioWordCue(surface="好", start_ms=88520, end_ms=88520),
            local_jp_renderer.AudioWordCue(surface="き！'", start_ms=88520, end_ms=88840),
        ],
        line_end_ms=90000,
    )

    assert stabilized == [
        local_jp_renderer.AudioWordCue(surface="好き！'", start_ms=88520, end_ms=88840),
    ]


def test_render_local_jp_video_emits_generated_lrc_for_txt_input(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.mp3"
    background_path = tmp_path / "background.png"
    bridge_lrc_path = tmp_path / "bridge.lrc"
    lyrics_txt_path = tmp_path / "lyrics.txt"
    output_dir = tmp_path / "out"
    font_path = tmp_path / "font.otf"

    audio_path.write_bytes(b"audio")
    background_path.write_bytes(b"background")
    bridge_lrc_path.write_text("[00:03.00]君を見つめた\n[00:05.50]好きな気持ちで\n", encoding="utf-8")
    lyrics_txt_path.write_text("君を見つめた\n好きな気持ちで\n", encoding="utf-8")
    font_path.write_bytes(b"font")

    monkeypatch.setattr(local_jp_renderer, "_probe_duration_ms", lambda path: 20_000)
    monkeypatch.setattr(local_jp_renderer, "detect_font_path", lambda: font_path)
    monkeypatch.setattr(local_jp_renderer, "load_font_set", lambda path: object())
    monkeypatch.setattr(local_jp_renderer, "_font_family_name", lambda path: "Noto Sans CJK JP")

    def fake_write_ass_subtitles(*, timed_lines, output_path, resolution, fonts, font_family):
        output_path.write_text("ass\n", encoding="utf-8")
        return output_path

    def fake_write_timeline_debug(timed_lines, output_path):
        output_path.write_text("{}\n", encoding="utf-8")
        return output_path

    def fake_render_video(*, audio_path, background_path, ass_path, output_path, duration_ms, resolution, font_path):
        output_path.write_bytes(b"video")
        return output_path

    monkeypatch.setattr(local_jp_renderer, "write_ass_subtitles", fake_write_ass_subtitles)
    monkeypatch.setattr(local_jp_renderer, "write_timeline_debug", fake_write_timeline_debug)
    monkeypatch.setattr(local_jp_renderer, "render_video", fake_render_video)
    monkeypatch.setattr(local_jp_renderer, "_align_timed_lines_to_audio", lambda timed_lines, audio_path, tokenizer=None: timed_lines)

    result = render_local_jp_video(
        audio_path=audio_path,
        background_path=background_path,
        output_dir=output_dir,
        output_stem="sample",
        lyrics_txt_path=lyrics_txt_path,
        txt_timing_bridge_lrc_path=bridge_lrc_path,
    )

    assert result.txt_path == output_dir / "sample.txt"
    assert result.txt_path.exists()
    assert result.txt_path.read_text(encoding="utf-8").splitlines() == [
        "君を見つめた",
        "好きな気持ちで",
    ]
    assert result.lrc_path == output_dir / "sample.lrc"
    assert result.lrc_path.exists()
    assert result.lrc_path.read_text(encoding="utf-8").splitlines() == [
        "[00:03.00]君を見つめた",
        "[00:05.50]好きな気持ちで",
    ]


def test_render_local_jp_video_retains_canonical_txt_and_lrc_for_lrc_input(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.mp3"
    background_path = tmp_path / "background.png"
    timing_lrc_path = tmp_path / "lyrics.lrc"
    output_dir = tmp_path / "out"
    font_path = tmp_path / "font.otf"

    audio_path.write_bytes(b"audio")
    background_path.write_bytes(b"background")
    timing_lrc_path.write_text(
        "\n".join(
            [
                "[00:03.00]君を見つめた",
                "[00:05.50]好きな気持ちで",
                "@Ruby1=君,きみ",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    font_path.write_bytes(b"font")

    monkeypatch.setattr(local_jp_renderer, "_probe_duration_ms", lambda path: 20_000)
    monkeypatch.setattr(local_jp_renderer, "detect_font_path", lambda: font_path)
    monkeypatch.setattr(local_jp_renderer, "load_font_set", lambda path: object())
    monkeypatch.setattr(local_jp_renderer, "_font_family_name", lambda path: "Noto Sans CJK JP")

    def fake_write_ass_subtitles(*, timed_lines, output_path, resolution, fonts, font_family):
        output_path.write_text("ass\n", encoding="utf-8")
        return output_path

    def fake_write_timeline_debug(timed_lines, output_path):
        output_path.write_text("{}\n", encoding="utf-8")
        return output_path

    def fake_render_video(*, audio_path, background_path, ass_path, output_path, duration_ms, resolution, font_path):
        output_path.write_bytes(b"video")
        return output_path

    monkeypatch.setattr(local_jp_renderer, "write_ass_subtitles", fake_write_ass_subtitles)
    monkeypatch.setattr(local_jp_renderer, "write_timeline_debug", fake_write_timeline_debug)
    monkeypatch.setattr(local_jp_renderer, "render_video", fake_render_video)
    monkeypatch.setattr(local_jp_renderer, "_align_timed_lines_to_audio", lambda timed_lines, audio_path, tokenizer=None: timed_lines)

    result = render_local_jp_video(
        audio_path=audio_path,
        background_path=background_path,
        output_dir=output_dir,
        output_stem="sample",
        timing_lrc_path=timing_lrc_path,
    )

    assert result.txt_path == output_dir / "sample.txt"
    assert result.txt_path.exists()
    assert result.txt_path.read_text(encoding="utf-8").splitlines() == [
        "君を見つめた",
        "好きな気持ちで",
    ]
    assert result.lrc_path == output_dir / "sample.lrc"
    assert result.lrc_path.exists()
    assert result.lrc_path.read_text(encoding="utf-8").splitlines() == [
        "[00:03.00]君を見つめた",
        "[00:05.50]好きな気持ちで",
    ]
