"""
Text-to-Audio Generator using Google Text-to-Speech (gTTS)
Converts text captions to MP3 audio files
"""
import os
import logging
from gtts import gTTS
import hashlib
import subprocess



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_audio(text: str, output_path: str, lang: str = 'en') -> bool:
    """
    Generate an audio file from text using Google Text-to-Speech
    
    Args:
        text: The text to convert to speech
        output_path: Full path where the audio file should be saved
        lang: Language code (default: 'en' for English)
        
    Returns:
        True if successful, False otherwise
    """
    try:
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
    Generate a unique audio filename based on image, text, and emotion.
    This makes sure different modes/emotions create different files.
    """
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    key = f"{image_filename}|{text}|{emotion}"
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"{base_name}_{emotion}_{hash_suffix}.mp3"


def apply_emotion_with_ffmpeg(audio_path: str, emotion: str) -> None:
    """
    Use ffmpeg to roughly shape the audio to match the selected emotion.
    This does not require pydub or audioop.
    """
    if emotion == "calm":
        return  # no change

    tmp_path = audio_path + ".tmp.mp3"

    # Choose filters per emotion
    if emotion == "energetic":
        # Faster + louder
        filter_str = "atempo=1.2,volume=1.5"
    elif emotion == "sad":
        # Slightly slower + softer + darker
        filter_str = "atempo=0.9,volume=0.8,lowpass=f=3000"
    elif emotion == "robotic":
        # Echo-y / metallic vibe
        filter_str = "aecho=0.8:0.88:60:0.4,acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1"
    elif emotion == "whisper":
        # Quieter + filtered
        filter_str = "volume=0.3,highpass=f=300,lowpass=f=6000"
    else:
        # Unknown emotion: do nothing
        return

    cmd = [
        "ffmpeg",
        "-y",              # overwrite
        "-i", audio_path,  # input
        "-af", filter_str, # audio filters
        tmp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp_path, audio_path)
        logger.info(f"Applied emotion '{emotion}' to audio: {audio_path}")
    except Exception as e:
        logger.error(f"Error applying emotion with ffmpeg: {str(e)}")
        # Clean up tmp file if something went wrong
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_audio(text: str, output_path: str, emotion: str = "calm") -> bool:
    """
    Basic TTS audio generation using gTTS, then optional ffmpeg shaping.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        apply_emotion_with_ffmpeg(output_path, emotion)
        logger.info(f"Audio saved to: {output_path} (emotion={emotion})")
        return True
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return False


def text_to_audio(text: str, audio_folder: str, image_filename: str, emotion: str = "calm") -> str | None:
    """
    Convert text to an audio file and return its path.
    'emotion' is used in the filename so changing modes/emotions
    forces a new file and avoids browser caching.
    """
    try:
        audio_filename = get_audio_filename(image_filename, text, emotion)
        audio_path = os.path.join(audio_folder, audio_filename)

        if generate_audio(text, audio_path, emotion):
            return audio_path
        else:
            return None

    except Exception as e:
        logger.error(f"Error in text_to_audio: {str(e)}")
        return None




