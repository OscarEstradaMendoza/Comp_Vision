"""
Text-to-Audio Generator Module

This module handles the conversion of text captions into audio files with various emotional styles.
It uses Google Text-to-Speech (gTTS) for initial speech synthesis and ffmpeg for post-processing
to apply emotional effects to the audio.

Key Features:
- Converts text to MP3 audio files using gTTS
- Applies emotional audio effects using ffmpeg filters
- Generates unique filenames to prevent browser caching issues
- Supports multiple emotion styles: calm, energetic, sad, robotic, whisper

Dependencies:
- gtts: Google Text-to-Speech library for speech synthesis
- ffmpeg: Audio processing tool for applying emotional effects
- hashlib: For generating unique filenames
- subprocess: For executing ffmpeg commands
"""

import os
import logging
from gtts import gTTS
import hashlib
import subprocess


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_audio(text: str, output_path: str, lang: str = 'en') -> bool:
    """
    Generate an audio file from text using Google Text-to-Speech.
    
    This is a legacy function that may not be used in the current implementation.
    The main audio generation is handled by the generate_audio() function below
    which includes emotion processing.
    
    Args:
        text (str): The text to convert to speech
        output_path (str): Full path where the audio file should be saved
        lang (str): Language code (default: 'en' for English)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input text
        if not text or not text.strip():
            logger.warning("Empty text provided for audio generation")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate audio using gTTS
        logger.info(f"Generating audio for text: {text[:50]}...")
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save audio file
        tts.save(output_path)
        logger.info(f"Audio file saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return False


def get_audio_filename(image_filename: str, text: str, emotion: str) -> str:
    """
    Generate a unique audio filename based on image filename, text content, and emotion.
    
    This function creates unique filenames to prevent browser caching issues when
    the same image is processed with different modes or emotions. The filename
    includes a hash of the image filename, text, and emotion to ensure uniqueness.
    
    Args:
        image_filename (str): Original image filename (e.g., "photo.jpg")
        text (str): The caption text (used in hash for uniqueness)
        emotion (str): Selected emotion style (used in hash and filename)
    
    Returns:
        str: Unique audio filename in format: "{base_name}_{emotion}_{hash}.mp3"
    
    Example:
        >>> get_audio_filename("image.jpg", "A cat", "energetic")
        "image_energetic_a1b2c3d4.mp3"
    """
    # Extract base name without extension from image filename
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    
    # Create a unique key combining all relevant information
    # This ensures different modes/emotions create different files
    key = f"{image_filename}|{text}|{emotion}"
    
    # Generate a short hash (8 characters) for uniqueness
    # MD5 is used here for speed, not security
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    
    # Return filename with emotion and hash
    return f"{base_name}_{emotion}_{hash_suffix}.mp3"


def apply_emotion_with_ffmpeg(audio_path: str, emotion: str) -> None:
    """
    Apply emotional effects to an audio file using ffmpeg audio filters.
    
    This function post-processes the generated TTS audio to match the selected emotion
    by applying various audio effects:
    - Energetic: Faster tempo and increased volume
    - Sad: Slower tempo, reduced volume, and low-pass filter for darker tone
    - Robotic: Echo and bit-crushing effects for metallic/robotic sound
    - Whisper: Reduced volume with high-pass and low-pass filters
    - Calm: No modifications (returns early)
    
    Args:
        audio_path (str): Path to the input audio file (will be overwritten)
        emotion (str): Emotion style to apply. Valid options:
            - 'calm': No changes (returns immediately)
            - 'energetic': Faster and louder
            - 'sad': Slower, softer, darker
            - 'robotic': Echo and bit-crushing effects
            - 'whisper': Quiet with filtered frequencies
    
    Note:
        This function requires ffmpeg to be installed on the system.
        If ffmpeg is not available or the command fails, the original audio
        file is preserved and an error is logged.
    
    Raises:
        No exceptions are raised - errors are logged and handled gracefully
    """
    # Calm emotion requires no processing
    if emotion == "calm":
        return  # no change

    # Create temporary file path for processing
    # The original file will be replaced after successful processing
    tmp_path = audio_path + ".tmp.mp3"

    # ========================================================================
    # EMOTION-SPECIFIC AUDIO FILTERS
    # ========================================================================
    # Each emotion uses different ffmpeg filter combinations
    
    if emotion == "energetic":
        # Faster tempo (1.2x speed) and increased volume (1.5x)
        # Creates an upbeat, energetic feel
        filter_str = "atempo=1.2,volume=1.5"
        
    elif emotion == "sad":
        # Slower tempo (0.9x speed), reduced volume (0.8x), and low-pass filter
        # The low-pass filter removes high frequencies for a darker, somber tone
        filter_str = "atempo=0.9,volume=0.8,lowpass=f=3000"
        
    elif emotion == "robotic":
        # Echo effect and bit-crushing for metallic/robotic sound
        # aecho: creates echo effect (in_gain:out_gain:delay:decay)
        # acrusher: reduces bit depth for robotic distortion
        filter_str = "aecho=0.8:0.88:60:0.4,acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1"
        
    elif emotion == "whisper":
        # Reduced volume (0.3x) with frequency filtering
        # highpass: removes low frequencies
        # lowpass: removes high frequencies
        # Creates a whisper-like, filtered sound
        filter_str = "volume=0.3,highpass=f=300,lowpass=f=6000"
        
    else:
        # Unknown emotion: do nothing
        logger.warning(f"Unknown emotion '{emotion}', skipping audio processing")
        return

    # ========================================================================
    # EXECUTE FFMPEG COMMAND
    # ========================================================================
    # Build ffmpeg command to apply audio filters
    cmd = [
        "ffmpeg",
        "-y",              # Overwrite output file without asking
        "-i", audio_path,  # Input audio file
        "-af", filter_str, # Apply audio filter chain
        tmp_path           # Output to temporary file
    ]

    try:
        # Run ffmpeg command
        # stdout and stderr are suppressed to avoid cluttering logs
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Replace original file with processed version
        # This is atomic on most systems
        os.replace(tmp_path, audio_path)
        logger.info(f"Applied emotion '{emotion}' to audio: {audio_path}")
        
    except Exception as e:
        # Log error but don't crash - original audio file is preserved
        logger.error(f"Error applying emotion with ffmpeg: {str(e)}")
        
        # Clean up temporary file if it exists
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_audio(text: str, output_path: str, emotion: str = "calm") -> bool:
    """
    Generate audio file from text using gTTS and apply emotional effects.
    
    This is the main audio generation function that:
    1. Converts text to speech using Google Text-to-Speech
    2. Saves the initial audio file
    3. Applies emotional effects using ffmpeg (if emotion is not 'calm')
    
    Args:
        text (str): The text to convert to speech
        output_path (str): Full path where the audio file should be saved
        emotion (str): Emotion style to apply (default: 'calm')
            Valid options: 'calm', 'energetic', 'sad', 'robotic', 'whisper'
    
    Returns:
        bool: True if audio was generated successfully, False otherwise
    
    Note:
        This function requires:
        - Internet connection (for gTTS API calls)
        - ffmpeg installed (for emotion processing, except 'calm')
    
    Example:
        >>> generate_audio("Hello world", "/path/to/audio.mp3", "energetic")
        True
    """
    try:
        # Step 1: Generate initial audio using Google Text-to-Speech
        # gTTS creates natural-sounding speech from text
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        
        # Step 2: Apply emotional effects using ffmpeg
        # This modifies the audio to match the selected emotion
        apply_emotion_with_ffmpeg(output_path, emotion)
        
        logger.info(f"Audio saved to: {output_path} (emotion={emotion})")
        return True
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return False


def text_to_audio(text: str, audio_folder: str, image_filename: str, emotion: str = "calm") -> str | None:
    """
    Main entry point for converting text to audio file.
    
    This function orchestrates the entire audio generation process:
    1. Generates a unique filename based on image, text, and emotion
    2. Creates the full file path
    3. Generates the audio file with emotional effects
    4. Returns the path to the generated audio file
    
    The unique filename ensures that:
    - Different modes/emotions create different files (prevents caching)
    - Same image with same settings can reuse cached audio
    - Browser caching works correctly for repeated requests
    
    Args:
        text (str): The caption text to convert to speech
        audio_folder (str): Directory where audio files should be saved
        image_filename (str): Original image filename (used for naming)
        emotion (str): Emotion style for audio (default: 'calm')
            Valid options: 'calm', 'energetic', 'sad', 'robotic', 'whisper'
    
    Returns:
        str | None: Path to the generated audio file if successful, None otherwise
    
    Example:
        >>> text_to_audio("A cat", "/static/audio", "photo.jpg", "energetic")
        "/static/audio/photo_energetic_a1b2c3d4.mp3"
    """
    try:
        # Generate unique filename based on all relevant parameters
        audio_filename = get_audio_filename(image_filename, text, emotion)
        
        # Construct full path to audio file
        audio_path = os.path.join(audio_folder, audio_filename)

        # Generate audio file with emotional effects
        if generate_audio(text, audio_path, emotion):
            return audio_path
        else:
            return None

    except Exception as e:
        logger.error(f"Error in text_to_audio: {str(e)}")
        return None
