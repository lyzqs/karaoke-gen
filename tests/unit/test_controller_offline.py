import logging
from unittest.mock import MagicMock

from karaoke_gen.lyrics_transcriber.core.config import LyricsConfig, OutputConfig, TranscriberConfig
from karaoke_gen.lyrics_transcriber.core.controller import LyricsTranscriber


def test_process_fetches_local_lyrics_without_artist_title(tmp_path):
    """A local lyrics file should still be fetched when artist/title are missing."""
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text("hello from file\n", encoding="utf-8")

    output_config = OutputConfig(
        output_styles_json="",
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        fetch_lyrics=True,
        run_transcription=False,
        run_correction=False,
        enable_review=False,
        render_video=False,
        generate_cdg=False,
    )

    transcriber = LyricsTranscriber(
        audio_filepath=str(tmp_path / "input.wav"),
        artist=None,
        title=None,
        transcriber_config=TranscriberConfig(enable_local_whisper=False),
        lyrics_config=LyricsConfig(
            lyrics_file=str(lyrics_path),
            disable_online_sources=True,
        ),
        output_config=output_config,
        transcribers={"noop": {"instance": MagicMock(), "priority": 1}},
        lyrics_providers={"file": MagicMock()},
        corrector=MagicMock(),
        output_generator=MagicMock(),
        logger=MagicMock(spec=logging.Logger),
    )

    transcriber.fetch_lyrics = MagicMock()
    transcriber.generate_outputs = MagicMock()

    transcriber.process()

    transcriber.fetch_lyrics.assert_called_once()
