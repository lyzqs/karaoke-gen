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

    result = render_local_jp_video(
        audio_path=audio_path,
        background_path=background_path,
        output_dir=output_dir,
        output_stem="sample",
        lyrics_txt_path=lyrics_txt_path,
        txt_timing_bridge_lrc_path=bridge_lrc_path,
    )

    assert result.lrc_path == output_dir / "sample.lrc"
    assert result.lrc_path.exists()
    assert result.lrc_path.read_text(encoding="utf-8").splitlines() == [
        "[00:03.00]君を見つめた",
        "[00:05.50]好きな気持ちで",
    ]
