import os
import sys
import logging
import subprocess
import shutil


class VideoBackgroundProcessor:
    """
    Handles video background processing for karaoke videos.
    Responsible for scaling, looping/trimming, darkening, and subtitle rendering.
    """

    def __init__(self, logger, ffmpeg_base_command, target_resolution=(3840, 2160)):
        """
        Initialize the VideoBackgroundProcessor.

        Args:
            logger: Logger instance for output
            ffmpeg_base_command: Base ffmpeg command with common flags
            target_resolution: Output resolution as (width, height)
        """
        self.logger = logger
        self.ffmpeg_base_command = ffmpeg_base_command
        self.target_resolution = target_resolution

        # Detect and configure hardware acceleration
        self.nvenc_available = self.detect_nvenc_support()
        self.configure_hardware_acceleration()

    def detect_nvenc_support(self):
        """Detect if NVENC hardware encoding is available."""
        try:
            self.logger.info("🔍 Detecting NVENC hardware acceleration...")

            # Check for nvidia-smi (indicates NVIDIA driver presence)
            try:
                nvidia_smi_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if nvidia_smi_result.returncode == 0:
                    gpu_info = nvidia_smi_result.stdout.strip()
                    self.logger.info(f"  ✓ NVIDIA GPU detected: {gpu_info}")
                else:
                    self.logger.debug(f"nvidia-smi failed: {nvidia_smi_result.stderr}")
                    self.logger.info("  ✗ NVENC not available (no NVIDIA GPU)")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
                self.logger.debug(f"nvidia-smi not available: {e}")
                self.logger.info("  ✗ NVENC not available (no NVIDIA GPU)")
                return False

            # Check for NVENC encoders in FFmpeg
            try:
                encoders_cmd = f"{self.ffmpeg_base_command} -hide_banner -encoders 2>/dev/null | grep nvenc"
                encoders_result = subprocess.run(encoders_cmd, shell=True, capture_output=True, text=True, timeout=10)
                if encoders_result.returncode == 0 and "nvenc" in encoders_result.stdout:
                    nvenc_encoders = [line.strip() for line in encoders_result.stdout.split("\n") if "nvenc" in line]
                    self.logger.debug(f"Found NVENC encoders: {nvenc_encoders}")
                else:
                    self.logger.debug("No NVENC encoders found in FFmpeg")
                    self.logger.info("  ✗ NVENC not available (no FFmpeg support)")
                    return False
            except Exception as e:
                self.logger.debug(f"Failed to check FFmpeg NVENC encoders: {e}")
                self.logger.info("  ✗ NVENC not available")
                return False

            # Check for libcuda.so.1 (critical for NVENC)
            try:
                libcuda_check = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10)
                if libcuda_check.returncode == 0 and "libcuda.so.1" in libcuda_check.stdout:
                    self.logger.debug("libcuda.so.1 found in system libraries")
                else:
                    self.logger.debug("libcuda.so.1 NOT found - may need nvidia/cuda:*-devel image")
                    self.logger.info("  ✗ NVENC not available (missing CUDA libraries)")
                    return False
            except Exception as e:
                self.logger.debug(f"Failed to check for libcuda.so.1: {e}")
                self.logger.info("  ✗ NVENC not available")
                return False

            # Test h264_nvenc encoder
            test_cmd = f"{self.ffmpeg_base_command} -hide_banner -loglevel error -f lavfi -i testsrc=duration=1:size=320x240:rate=1 -c:v h264_nvenc -f null -"
            self.logger.debug(f"Testing NVENC: {test_cmd}")

            try:
                result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    self.logger.info("  ✓ NVENC encoding available")
                    return True
                else:
                    self.logger.debug(f"NVENC test failed (exit code {result.returncode}): {result.stderr}")
                    self.logger.info("  ✗ NVENC not available")
                    return False

            except subprocess.TimeoutExpired:
                self.logger.debug("NVENC test timed out")
                self.logger.info("  ✗ NVENC not available (timeout)")
                return False

        except Exception as e:
            self.logger.debug(f"Failed to detect NVENC support: {e}")
            self.logger.info("  ✗ NVENC not available (error)")
            return False

    def configure_hardware_acceleration(self):
        """Configure hardware acceleration settings based on detected capabilities."""
        if self.nvenc_available:
            self.video_encoder = "h264_nvenc"
            self.hwaccel_decode_flags = "-hwaccel cuda"
            self.scale_filter = "scale"
            self.logger.info("🚀 Using NVENC hardware acceleration for video encoding")
        else:
            self.video_encoder = "libx264"
            self.hwaccel_decode_flags = ""
            self.scale_filter = "scale"
            self.logger.info("🔧 Using software encoding (libx264) for video")

    def get_nvenc_quality_settings(self):
        """Get NVENC settings for high quality encoding."""
        return "-preset p4 -tune hq -rc vbr -cq 18 -spatial-aq 1 -temporal-aq 1 -b:v 8000k -maxrate 15000k -bufsize 16000k"

    def get_audio_duration(self, audio_path):
        """
        Get duration of audio file in seconds using ffprobe.

        Args:
            audio_path: Path to audio file

        Returns:
            float: Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            self.logger.info(f"Audio duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            self.logger.error(f"Failed to get audio duration: {e}")
            raise

    def escape_filter_path(self, path):
        """
        Escape a file path for use in ffmpeg filter expressions.

        FFmpeg filter syntax uses single quotes to protect special characters
        like spaces, colons, and semicolons. Single quotes within the path
        are escaped using the '\\'' pattern (end quote, literal quote, start quote).

        Args:
            path: File path to escape

        Returns:
            str: Escaped path (wrapped in single quotes)
        """
        # Escape single quotes: ' becomes '\'' (end quote, \', start quote)
        escaped = path.replace("'", "'\\''")
        # Wrap entire path in single quotes
        return f"'{escaped}'"

    def build_video_filter(self, ass_subtitles_path, darkness_percent, fonts_dir=None):
        """
        Build the video filter chain for scaling, darkening, and subtitles.

        Args:
            ass_subtitles_path: Path to ASS subtitle file
            darkness_percent: Darkness overlay percentage (0-100)
            fonts_dir: Optional fonts directory for ASS rendering

        Returns:
            str: FFmpeg filter string
        """
        filters = []

        width, height = self.target_resolution

        # Scale to the target frame with intelligent cropping (not stretching).
        # force_original_aspect_ratio=increase ensures we scale up to fill the frame
        # then crop to the exact target dimensions.
        filters.append(
            f"scale=w={width}:h={height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        )

        # Add darkening overlay if requested (before ASS subtitles)
        if darkness_percent > 0:
            # Convert percentage (0-100) to alpha (0.0-1.0)
            alpha = darkness_percent / 100.0
            filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black@{alpha:.2f}:t=fill")

        # Add ASS subtitle filter
        ass_escaped = self.escape_filter_path(ass_subtitles_path)
        ass_filter = f"ass={ass_escaped}"

        # Add fonts directory if provided
        if fonts_dir and os.path.isdir(fonts_dir):
            fonts_escaped = self.escape_filter_path(fonts_dir)
            ass_filter += f":fontsdir={fonts_escaped}"

        filters.append(ass_filter)

        # Combine all filters with commas
        return ",".join(filters)

    def execute_command_with_fallback(self, gpu_command, cpu_command, description):
        """
        Execute GPU command with automatic fallback to CPU if it fails.

        Args:
            gpu_command: Command to try with GPU acceleration
            cpu_command: Fallback command for CPU encoding
            description: Description for logging

        Raises:
            Exception: If both GPU and CPU commands fail
        """
        self.logger.info(f"{description}")

        # Try GPU-accelerated command first if available
        if self.nvenc_available and gpu_command != cpu_command:
            self.logger.debug(f"Attempting hardware-accelerated encoding: {gpu_command}")
            try:
                result = subprocess.run(gpu_command, shell=True, capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    self.logger.info(f"✓ Hardware acceleration successful")
                    return
                else:
                    self.logger.warning(f"✗ Hardware acceleration failed (exit code {result.returncode})")
                    self.logger.warning(f"GPU Command: {gpu_command}")

                    if result.stderr:
                        self.logger.warning(f"FFmpeg STDERR: {result.stderr}")
                    if result.stdout:
                        self.logger.warning(f"FFmpeg STDOUT: {result.stdout}")
                    self.logger.info("Falling back to software encoding...")

            except subprocess.TimeoutExpired:
                self.logger.warning("✗ Hardware acceleration timed out, falling back to software encoding")
            except Exception as e:
                self.logger.warning(f"✗ Hardware acceleration failed with exception: {e}, falling back to software encoding")

        # Use CPU command (either as fallback or primary method)
        self.logger.debug(f"Running software encoding: {cpu_command}")
        try:
            result = subprocess.run(cpu_command, shell=True, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                error_msg = f"Software encoding failed with exit code {result.returncode}"
                self.logger.error(error_msg)
                self.logger.error(f"CPU Command: {cpu_command}")
                if result.stderr:
                    self.logger.error(f"FFmpeg STDERR: {result.stderr}")
                if result.stdout:
                    self.logger.error(f"FFmpeg STDOUT: {result.stdout}")
                raise Exception(f"{error_msg}: {cpu_command}")
            else:
                self.logger.info(f"✓ Software encoding successful")

        except subprocess.TimeoutExpired:
            error_msg = "Software encoding timed out"
            self.logger.error(error_msg)
            raise Exception(f"{error_msg}: {cpu_command}")
        except Exception as e:
            if "Software encoding failed" not in str(e):
                error_msg = f"Software encoding failed with exception: {e}"
                self.logger.error(error_msg)
                raise Exception(f"{error_msg}: {cpu_command}")
            else:
                raise

    def process_video_background(
        self, video_path, audio_path, ass_subtitles_path, output_path, darkness_percent=0, audio_duration=None
    ):
        """
        Process video background with scaling, looping/trimming, darkening, and subtitle rendering.

        Args:
            video_path: Path to input video file
            audio_path: Path to audio file (used for duration and audio track)
            ass_subtitles_path: Path to ASS subtitle file
            output_path: Path to output video file
            darkness_percent: Darkness overlay percentage (0-100), default 0
            audio_duration: Optional pre-calculated audio duration (will calculate if not provided)

        Returns:
            str: Path to output file

        Raises:
            Exception: If video processing fails
        """
        self.logger.info(f"Processing video background: {video_path}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Darkness: {darkness_percent}%")

        # Validate inputs
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.isfile(ass_subtitles_path):
            raise FileNotFoundError(f"ASS subtitle file not found: {ass_subtitles_path}")

        # Validate darkness parameter
        if not 0 <= darkness_percent <= 100:
            raise ValueError(f"Darkness percentage must be between 0 and 100, got {darkness_percent}")

        # Get audio duration if not provided
        if audio_duration is None:
            audio_duration = self.get_audio_duration(audio_path)

        # Check for optional fonts directory (matching video.py behavior)
        fonts_dir = os.environ.get("KARAOKE_FONTS_DIR")

        # Build video filter chain
        vf_filter = self.build_video_filter(ass_subtitles_path, darkness_percent, fonts_dir)

        # Build commands for GPU and CPU encoding
        # Use -stream_loop -1 to loop video indefinitely, -shortest to cut when audio ends
        base_inputs = f'-stream_loop -1 -i "{video_path}" -i "{audio_path}"'

        # GPU-accelerated version
        gpu_command = (
            f"{self.ffmpeg_base_command} {self.hwaccel_decode_flags} {base_inputs} "
            f'-c:a flac -vf "{vf_filter}" -c:v {self.video_encoder} '
            f"{self.get_nvenc_quality_settings()} -shortest -y \"{output_path}\""
        )

        # Software fallback version
        cpu_command = (
            f'{self.ffmpeg_base_command} {base_inputs} '
            f'-c:a flac -vf "{vf_filter}" -c:v libx264 -preset fast '
            f"-b:v 5000k -minrate 5000k -maxrate 20000k -bufsize 10000k "
            f'-shortest -y "{output_path}"'
        )

        # Execute with fallback
        self.execute_command_with_fallback(
            gpu_command, cpu_command, f"Rendering video with background, subtitles, and effects"
        )

        # Verify output was created
        if not os.path.isfile(output_path):
            raise Exception(f"Output video file was not created: {output_path}")

        self.logger.info(f"✓ Video background processing complete: {output_path}")
        return output_path
