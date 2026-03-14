import logging
from pathlib import Path
from unittest.mock import patch

from karaoke_gen.lyrics_transcriber.lyrics.base_lyrics_provider import LyricsProviderConfig
from karaoke_gen.lyrics_transcriber.lyrics.file_provider import FileProvider


def _create_provider(lyrics_file: Path) -> FileProvider:
    logger = logging.getLogger(f"test.file_provider.{lyrics_file.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    return FileProvider(LyricsProviderConfig(lyrics_file=str(lyrics_file)), logger=logger)


def test_fetch_data_from_source_normalizes_lrc_reference_lyrics(tmp_path):
    lyrics_file = tmp_path / "reference.lrc"
    lyrics_file.write_text(
        "[ar:Test Artist]\n"
        "[ti:Test Song]\n"
        "[00:01.00]Hello <00:01.20>world\n"
        "[00:02.00][00:03.50]Second line\n",
        encoding="utf-8",
    )
    provider = _create_provider(lyrics_file)

    with patch("karaoke_gen.lyrics_transcriber.lyrics.file_provider.KaraokeLyricsProcessor") as mock_processor_cls:
        mock_processor_cls.return_value.process.return_value = "Hello world\nSecond line"

        result = provider._fetch_data_from_source("Test Artist", "Test Song")

    assert result == {
        "text": "Hello world\nSecond line",
        "source": "file",
        "filepath": str(lyrics_file),
    }

    processor_kwargs = mock_processor_cls.call_args.kwargs
    assert processor_kwargs["input_lyrics_text"] == "Hello world\nSecond line"
    assert "input_filename" not in processor_kwargs


def test_fetch_data_from_source_uses_filename_for_non_lrc_files(tmp_path):
    lyrics_file = tmp_path / "reference.txt"
    lyrics_file.write_text("Hello world", encoding="utf-8")
    provider = _create_provider(lyrics_file)

    with patch("karaoke_gen.lyrics_transcriber.lyrics.file_provider.KaraokeLyricsProcessor") as mock_processor_cls:
        mock_processor_cls.return_value.process.return_value = "Hello world"

        result = provider._fetch_data_from_source("Test Artist", "Test Song")

    assert result == {
        "text": "Hello world",
        "source": "file",
        "filepath": str(lyrics_file),
    }

    processor_kwargs = mock_processor_cls.call_args.kwargs
    assert processor_kwargs["input_filename"] == str(lyrics_file)
    assert "input_lyrics_text" not in processor_kwargs


def test_convert_result_format_includes_lrc_ruby_annotations(tmp_path):
    lyrics_file = tmp_path / "reference.lrc"
    lyrics_file.write_text(
        "[00:01.00]君を見つめた\n"
        "@Ruby2=見,み\n"
        "@Ruby1=君,きみ\n",
        encoding="utf-8",
    )
    provider = _create_provider(lyrics_file)
    provider.title = "Test Song"
    provider.artist = "Test Artist"

    lyrics_data = provider._convert_result_format(
        {
            "text": "君を見つめた",
            "source": "file",
            "filepath": str(lyrics_file),
        }
    )

    assert lyrics_data.metadata.provider_metadata["ruby_annotations"] == [
        {"index": 1, "base_text": "君", "ruby_text": "きみ"},
        {"index": 2, "base_text": "見", "ruby_text": "み"},
    ]
