from dataclasses import dataclass
import os
import logging
from typing import List, Optional
import json
from copy import deepcopy

from karaoke_gen.lyrics_transcriber.types import LyricsData, LyricsSegment, Word
from karaoke_gen.lyrics_transcriber.correction.corrector import CorrectionResult
from karaoke_gen.lyrics_transcriber.output.plain_text import PlainTextGenerator
from karaoke_gen.lyrics_transcriber.ruby import deserialize_ruby_annotations
from karaoke_gen.lyrics_transcriber.output.lyrics_file import LyricsFileGenerator
from karaoke_gen.lyrics_transcriber.output.subtitles import SubtitlesGenerator
from karaoke_gen.lyrics_transcriber.output.video import VideoGenerator
from karaoke_gen.lyrics_transcriber.output.segment_resizer import SegmentResizer
from karaoke_gen.lyrics_transcriber.output.cdg import CDGGenerator
from karaoke_gen.lyrics_transcriber.output.countdown_processor import CountdownProcessor
from karaoke_gen.lyrics_transcriber.core.config import OutputConfig


@dataclass
class OutputPaths:
    """Holds paths for generated output files."""

    lrc: Optional[str] = None
    ass: Optional[str] = None
    video: Optional[str] = None
    original_txt: Optional[str] = None
    corrected_txt: Optional[str] = None
    corrections_json: Optional[str] = None
    cdg: Optional[str] = None
    mp3: Optional[str] = None
    cdg_zip: Optional[str] = None


class OutputGenerator:
    """Handles generation of various lyrics output formats."""

    def __init__(
        self,
        config: OutputConfig,
        logger: Optional[logging.Logger] = None,
        preview_mode: bool = False,
    ):
        """
        Initialize OutputGenerator with configuration.

        Args:
            config: OutputConfig instance with required paths and settings
            logger: Optional logger instance
            preview_mode: Boolean indicating if the generator is in preview mode
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Initializing OutputGenerator with config: {self.config}")

        # Load output styles from JSON if provided, otherwise use defaults
        if self.config.output_styles_json and os.path.exists(self.config.output_styles_json):
            try:
                with open(self.config.output_styles_json, "r") as f:
                    self.config.styles = json.load(f)
                self.logger.debug(f"Loaded output styles from: {self.config.output_styles_json}")
            except Exception as e:
                if self.config.render_video or self.config.generate_cdg:
                    # Only raise error for video/CDG since they require styles
                    raise ValueError(f"Failed to load output styles file: {str(e)}")
                else:
                    # For other outputs, just log warning and continue with empty styles
                    self.logger.warning(f"Failed to load output styles file: {str(e)}")
                    self.config.styles = {}
        else:
            # No styles file provided or doesn't exist - use defaults
            if self.config.render_video or self.config.generate_cdg:
                self.logger.info("No output styles file provided, using default karaoke styles")
                self.config.styles = self._get_default_styles()
            else:
                self.config.styles = {}

        # Set video resolution parameters
        self.video_resolution_num, self.font_size, self.line_height = self._get_video_params(self.config.video_resolution)
        self.logger.info(f"Video resolution: {self.video_resolution_num}, font size: {self.font_size}, line height: {self.line_height}")

        # Initialize generators
        self.plain_text = PlainTextGenerator(self.config.output_dir, self.logger)
        self.lyrics_file = LyricsFileGenerator(self.config.output_dir, self.logger)

        if self.config.generate_cdg:
            self.cdg = CDGGenerator(self.config.output_dir, self.logger)

        self.preview_mode = preview_mode
        if self.config.render_video:
            # Apply preview mode scaling if needed
            if self.preview_mode:
                # Scale down from 4K (2160p) to 360p - factor of 1/6
                scale_factor = 1 / 6

                # Scale down top padding for preview if it exists
                if "karaoke" in self.config.styles and "top_padding" in self.config.styles["karaoke"]:
                    self.logger.info(f"Preview mode: Found top_padding: {self.config.styles['karaoke']['top_padding']}")
                    original_padding = self.config.styles["karaoke"]["top_padding"]
                    if original_padding is not None:
                        # Scale down from 4K (2160p) to 360p - factor of 1/6
                        self.config.styles["karaoke"]["top_padding"] = original_padding * scale_factor
                        self.logger.info(f"Preview mode: Scaled down top_padding to: {self.config.styles['karaoke']['top_padding']}")

                # Scale down font size for preview if it exists
                if "karaoke" in self.config.styles and "font_size" in self.config.styles["karaoke"]:
                    self.logger.info(f"Preview mode: Found font_size: {self.config.styles['karaoke']['font_size']}")
                    original_font_size = self.config.styles["karaoke"]["font_size"]
                    if original_font_size is not None:
                        # Scale down from 4K (2160p) to 360p - factor of 1/6
                        self.font_size = original_font_size * scale_factor
                        self.config.styles["karaoke"]["font_size"] = self.font_size
                        self.logger.info(f"Preview mode: Scaled down font_size to: {self.font_size}")

        # Get max_line_length from styles if available, otherwise use config default
        max_line_length = self.config.styles.get("karaoke", {}).get("max_line_length", self.config.default_max_line_length)
        self.logger.info(f"Using max_line_length: {max_line_length}")
        self.segment_resizer = SegmentResizer(max_line_length=max_line_length, logger=self.logger)

        if self.config.render_video:
            # Initialize subtitle generator with potentially scaled values
            self.subtitle = SubtitlesGenerator(
                output_dir=self.config.output_dir,
                video_resolution=self.video_resolution_num,
                font_size=self.font_size,
                line_height=self.line_height,
                styles=self.config.styles,
                subtitle_offset_ms=self.config.subtitle_offset_ms,
                logger=self.logger,
            )

            self.video = VideoGenerator(
                output_dir=self.config.output_dir,
                cache_dir=self.config.cache_dir,
                video_resolution=self.video_resolution_num,
                styles=self.config.styles,
                logger=self.logger,
            )

        # Log the configured directories
        self.logger.debug(f"Initialized OutputGenerator with output_dir: {self.config.output_dir}")
        self.logger.debug(f"Using cache_dir: {self.config.cache_dir}")

    def _get_ruby_annotations(
        self,
        transcription_corrected: Optional[CorrectionResult],
        lyrics_results: dict[str, LyricsData],
    ):
        """Return ruby annotations from the file lyrics provider, if available."""
        candidate_sets = []

        if transcription_corrected and transcription_corrected.reference_lyrics:
            file_reference = transcription_corrected.reference_lyrics.get("file")
            if file_reference:
                candidate_sets.append(file_reference)
            candidate_sets.extend(transcription_corrected.reference_lyrics.values())

        if lyrics_results:
            file_reference = lyrics_results.get("file")
            if file_reference:
                candidate_sets.append(file_reference)
            candidate_sets.extend(lyrics_results.values())

        for lyrics_data in candidate_sets:
            if not lyrics_data:
                continue

            provider_metadata = getattr(lyrics_data.metadata, "provider_metadata", {}) or {}
            ruby_payload = provider_metadata.get("ruby_annotations")
            if not ruby_payload:
                continue

            ruby_annotations = deserialize_ruby_annotations(ruby_payload)
            if ruby_annotations:
                self.logger.info("Using %d ruby annotations from lyrics provider '%s'", len(ruby_annotations), lyrics_data.source)
                return ruby_annotations

        return []

    def _strip_countdown_segments(self, segments: List[LyricsSegment]) -> List[LyricsSegment]:
        """Remove visible countdown lyric segments from final output."""
        filtered = [
            segment
            for segment in segments
            if segment.text.strip() != CountdownProcessor.COUNTDOWN_TEXT
        ]
        removed = len(segments) - len(filtered)
        if removed:
            self.logger.info("Removed %d countdown segment(s) from final output", removed)
        return filtered

    def _copy_word_with_timing(self, source_word: Word, timed_word: Word) -> Word:
        """Return a canonical word with the timing from an existing timed word."""
        return Word(
            id=timed_word.id,
            text=source_word.text,
            start_time=timed_word.start_time,
            end_time=timed_word.end_time,
            confidence=timed_word.confidence,
            created_during_correction=False,
        )

    def _distribute_reference_words_over_segment(
        self,
        reference_segment: LyricsSegment,
        timed_segment: LyricsSegment,
    ) -> LyricsSegment:
        """Fallback: keep canonical text and evenly distribute word timings across a segment."""
        if not reference_segment.words:
            return LyricsSegment(
                id=timed_segment.id,
                text=reference_segment.text.strip(),
                words=[],
                start_time=timed_segment.start_time,
                end_time=timed_segment.end_time,
            )

        start_time = timed_segment.start_time
        end_time = max(timed_segment.end_time, start_time)
        word_count = len(reference_segment.words)
        span = max(end_time - start_time, 0.0)
        step = span / word_count if word_count else 0.0

        rebuilt_words: List[Word] = []
        for index, ref_word in enumerate(reference_segment.words):
            word_start = start_time + (step * index)
            word_end = end_time if index == word_count - 1 else start_time + (step * (index + 1))
            rebuilt_words.append(
                Word(
                    id=timed_segment.words[min(index, len(timed_segment.words) - 1)].id if timed_segment.words else ref_word.id,
                    text=ref_word.text,
                    start_time=word_start,
                    end_time=word_end,
                    confidence=1.0,
                    created_during_correction=False,
                )
            )

        return LyricsSegment(
            id=timed_segment.id,
            text=reference_segment.text.strip(),
            words=rebuilt_words,
            start_time=rebuilt_words[0].start_time,
            end_time=rebuilt_words[-1].end_time,
        )

    def _rebuild_segments_from_reference_words(
        self,
        reference_segments: List[LyricsSegment],
        timed_words: List[Word],
    ) -> List[LyricsSegment]:
        """Rebuild canonical segments using an equal-length timed word sequence."""
        rebuilt_segments: List[LyricsSegment] = []
        timed_index = 0

        for reference_segment in reference_segments:
            word_count = len(reference_segment.words)
            timed_slice = timed_words[timed_index : timed_index + word_count]
            timed_index += word_count

            if not timed_slice:
                continue

            rebuilt_words = [
                self._copy_word_with_timing(reference_word, timed_word)
                for reference_word, timed_word in zip(reference_segment.words, timed_slice)
            ]
            rebuilt_segments.append(
                LyricsSegment(
                    id=reference_segment.id,
                    text=reference_segment.text.strip(),
                    words=rebuilt_words,
                    start_time=rebuilt_words[0].start_time,
                    end_time=rebuilt_words[-1].end_time,
                )
            )

        return rebuilt_segments

    def _remap_segments_to_reference(
        self,
        timed_segments: List[LyricsSegment],
        reference_segments: List[LyricsSegment],
    ) -> List[LyricsSegment]:
        """Prefer canonical reference text while preserving as much timing as possible."""
        if not timed_segments or not reference_segments:
            return timed_segments

        timed_words = [word for segment in timed_segments for word in segment.words]
        reference_words = [word for segment in reference_segments for word in segment.words]

        if reference_words and len(reference_words) == len(timed_words):
            self.logger.info(
                "Rebuilding final output from canonical reference lyrics (%d words)",
                len(reference_words),
            )
            rebuilt = self._rebuild_segments_from_reference_words(reference_segments, timed_words)
            if rebuilt:
                return rebuilt

        if len(reference_segments) == len(timed_segments):
            self.logger.info(
                "Canonical segment count matches timed output (%d segments); remapping segment-by-segment",
                len(reference_segments),
            )
            rebuilt_segments: List[LyricsSegment] = []
            for reference_segment, timed_segment in zip(reference_segments, timed_segments):
                if reference_segment.words and len(reference_segment.words) == len(timed_segment.words):
                    rebuilt_words = [
                        self._copy_word_with_timing(reference_word, timed_word)
                        for reference_word, timed_word in zip(reference_segment.words, timed_segment.words)
                    ]
                    rebuilt_segments.append(
                        LyricsSegment(
                            id=timed_segment.id,
                            text=reference_segment.text.strip(),
                            words=rebuilt_words,
                            start_time=rebuilt_words[0].start_time,
                            end_time=rebuilt_words[-1].end_time,
                        )
                    )
                else:
                    rebuilt_segments.append(
                        self._distribute_reference_words_over_segment(reference_segment, timed_segment)
                    )
            return rebuilt_segments

        if reference_words and len(reference_words) < len(timed_words):
            self.logger.warning(
                "Dropping %d extra timed word(s) to restore canonical output",
                len(timed_words) - len(reference_words),
            )
            rebuilt = self._rebuild_segments_from_reference_words(
                reference_segments,
                timed_words[: len(reference_words)],
            )
            if rebuilt:
                return rebuilt

        self.logger.warning(
            "Unable to remap final output to canonical reference lyrics cleanly; keeping reviewed segments"
        )
        return timed_segments

    def _prepare_correction_for_output(self, correction_result: CorrectionResult) -> CorrectionResult:
        """Apply offline-only sanitization before generating final artifacts."""
        prepared = deepcopy(correction_result)

        if self.config.strip_countdown_text:
            prepared.corrected_segments = self._strip_countdown_segments(prepared.corrected_segments)
            if prepared.resized_segments:
                prepared.resized_segments = self._strip_countdown_segments(prepared.resized_segments)

        canonical_source = self.config.prefer_reference_lyrics_source
        if canonical_source and prepared.reference_lyrics:
            reference_lyrics = prepared.reference_lyrics.get(canonical_source)
            if reference_lyrics and reference_lyrics.segments:
                prepared.corrected_segments = self._remap_segments_to_reference(
                    prepared.corrected_segments,
                    reference_lyrics.segments,
                )
                prepared.resized_segments = []

        return prepared

    def generate_outputs(
        self,
        transcription_corrected: Optional[CorrectionResult],
        lyrics_results: dict[str, LyricsData],
        output_prefix: str,
        audio_filepath: str,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        ass_only: bool = False,
    ) -> OutputPaths:
        """Generate all requested output formats.

        Args:
            transcription_corrected: Corrected transcription data
            lyrics_results: Lyrics data from various providers
            output_prefix: Prefix for output filenames
            audio_filepath: Path to audio file
            artist: Optional artist name
            title: Optional title
            ass_only: If True (only in preview_mode), generate only ASS subtitles
                      without video encoding. Useful when video encoding is offloaded
                      to an external service.
        """
        outputs = OutputPaths()

        try:
            # Only process transcription-related outputs if we have transcription data
            if transcription_corrected:
                transcription_corrected = self._prepare_correction_for_output(transcription_corrected)

                # Resize corrected segments
                resized_segments = self.segment_resizer.resize_segments(transcription_corrected.corrected_segments)
                transcription_corrected.resized_segments = resized_segments
                ruby_annotations = self._get_ruby_annotations(transcription_corrected, lyrics_results)

                # For preview, we only need to generate ASS and video
                if self.preview_mode:
                    # Generate ASS subtitles for preview
                    outputs.ass = self.subtitle.generate_ass(
                        transcription_corrected.resized_segments,
                        output_prefix,
                        audio_filepath,
                        ruby_annotations=ruby_annotations,
                    )

                    # Generate preview video (unless ass_only mode for GCE encoding)
                    if not ass_only:
                        outputs.video = self.video.generate_preview_video(outputs.ass, audio_filepath, output_prefix)

                    return outputs

                # Normal output generation (non-preview mode)
                # Generate plain lyrics files for each provider
                for name, lyrics_data in lyrics_results.items():
                    self.plain_text.write_lyrics(lyrics_data, output_prefix)

                # Write original (uncorrected) transcription
                outputs.original_txt = self.plain_text.write_original_transcription(transcription_corrected, output_prefix)

                outputs.corrections_json = self.write_corrections_data(transcription_corrected, output_prefix)

                # Write corrected lyrics as plain text
                outputs.corrected_txt = self.plain_text.write_corrected_lyrics(resized_segments, output_prefix)

                # Generate LRC using LyricsFileGenerator
                outputs.lrc = self.lyrics_file.generate_lrc(resized_segments, output_prefix)

                # Generate CDG file if requested
                if self.config.generate_cdg:
                    outputs.cdg, outputs.mp3, outputs.cdg_zip = self.cdg.generate_cdg(
                        segments=resized_segments,
                        audio_file=audio_filepath,
                        title=title or output_prefix,
                        artist=artist or "",
                        cdg_styles=self.config.styles["cdg"],
                    )

                # Generate video if requested
                if self.config.render_video:
                    # Generate ASS subtitles
                    outputs.ass = self.subtitle.generate_ass(
                        resized_segments,
                        output_prefix,
                        audio_filepath,
                        ruby_annotations=ruby_annotations,
                    )
                    outputs.video = self.video.generate_video(outputs.ass, audio_filepath, output_prefix)

            return outputs

        except Exception as e:
            self.logger.error(f"Failed to generate outputs: {str(e)}")
            raise

    def _get_output_path(self, output_prefix: str, extension: str) -> str:
        """Generate full output path for a file."""
        return os.path.join(self.config.output_dir or self.config.cache_dir, f"{output_prefix}.{extension}")

    def _get_video_params(self, resolution: str) -> tuple:
        """Get video parameters: (width, height), font_size, line_height based on video resolution config."""
        # Get resolution dimensions
        resolution_map = {
            "4k": (3840, 2160),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "360p": (640, 360),
        }

        if resolution not in resolution_map:
            raise ValueError("Invalid video_resolution value. Must be one of: 4k, 1080p, 720p, 360p")

        resolution_dims = resolution_map[resolution]

        # Default font sizes for each resolution
        default_font_sizes = {
            "4k": 250,
            "1080p": 120,
            "720p": 100,
            "360p": 40,
        }

        # Get font size from styles if available, otherwise use default
        font_size = self.config.styles.get("karaoke", {}).get("font_size", default_font_sizes[resolution])

        # Line height matches font size for all except 360p
        line_height = 50 if resolution == "360p" else font_size

        return resolution_dims, font_size, line_height

    def _get_default_styles(self) -> dict:
        """Get default styles for video/CDG generation when no styles file is provided."""
        return {
            "karaoke": {
                # Video background
                "background_color": "#000000",
                "background_image": None,
                # Font settings
                "font": "Arial",
                "font_path": "",  # Must be string, not None (for ASS generator)
                "ass_name": "Default",
                # Colors in "R, G, B, A" format (required by ASS)
                "primary_color": "112, 112, 247, 255",
                "secondary_color": "255, 255, 255, 255",
                "outline_color": "26, 58, 235, 255",
                "back_color": "0, 0, 0, 0",
                # Boolean style options
                "bold": False,
                "italic": False,
                "underline": False,
                "strike_out": False,
                # Numeric style options (all required for ASS)
                "scale_x": 100,
                "scale_y": 100,
                "spacing": 0,
                "angle": 0.0,
                "border_style": 1,
                "outline": 1,
                "shadow": 0,
                "margin_l": 0,
                "margin_r": 0,
                "margin_v": 0,
                "encoding": 0,
                # Layout settings
                "max_line_length": 40,
                "max_visible_lines": 2,
                "top_padding": 200,
                "bottom_padding_percent": 16.0,
                "line_left_padding_percent": 11.0,
                "line_right_padding_percent": 11.0,
                "show_section_markers": False,
                "font_size": 100,
            },
            "cdg": {
                "font_path": None,
                "instrumental_background": None,
                "title_screen_background": None,
                "outro_background": None,
            },
        }

    def write_corrections_data(self, correction_result: CorrectionResult, output_prefix: str) -> str:
        """Write corrections data to JSON file."""
        self.logger.info("Writing corrections data JSON")
        output_path = self._get_output_path(f"{output_prefix} (Lyrics Corrections)", "json")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(correction_result.to_dict(), f, indent=2, ensure_ascii=False)
            self.logger.info(f"Corrections data JSON generated: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to write corrections data JSON: {str(e)}")
            raise
