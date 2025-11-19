from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import logging
import random
from caption_model import get_caption_model, generate_caption
from audio_generator import text_to_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'cyberpunk-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def apply_mode_to_caption(caption: str, mode: str) -> str:
    """
    Take the base image caption and transform it depending on the selected mode.
    - caption: plain BLIP caption
    - mode: 'caption', 'story', 'poetic', or 'funny'
    """
    if not caption:
        return caption

    base = caption.strip().rstrip(".")
    # A slightly more natural version we can plug into text
    scene = base[0].lower() + base[1:] if len(base) > 1 else base

    if mode == "caption":
        variants = [
            base,
            f"{base}. A moment frozen in neon light.",
            f"{base}, framed by the distant hum of the city.",
            f"{base}. It looks like a still from a sci-fi movie.",
            f"{base}. You can almost hear the electricity in the air.",
        ]
        return random.choice(variants)

    if mode == "story":
        openings = [
            f"In a forgotten corner of the city, {scene}.",
            f"On a night when the neon refused to sleep, {scene}.",
            f"Somewhere between dream and glitch, {scene}.",
        ]
        middles = [
            " Rain slid down chrome and glass, turning every light into a streak of color.",
            " The streets hummed with drones, tired footsteps, and quiet conversations no one was meant to hear.",
            " Above, billboards argued in looping holograms while the sky pretended not to notice.",
        ]
        twists = [
            " Someone passing by felt the strange certainty that this moment mattered.",
            " A stranger stopped, saving the scene in their memory like a secret file.",
            " No one realized this tiny scene would start a chain of events that couldn’t be undone.",
        ]
        endings = [
            " By morning, the city would be exactly the same—except for one person who finally decided to change.",
            " When the sun finally pushed through the clouds, the world looked identical, but the story had already shifted.",
            " Long after the lights dimmed, the echo of that moment kept rewriting itself in the network’s memory.",
        ]

        story = random.choice(openings) + random.choice(middles) + \
                random.choice(twists) + " " + random.choice(endings)
        return story

    if mode == "poetic":
        templates = [
            f"{base}.\n"
            "Neon drips down the glass,\n"
            "Shadows lean into the rain,\n"
            "Circuits whisper under tired boots,\n"
            "And the city pretends not to dream.",

            f"{base}.\n"
            "Pixels blur into midnight,\n"
            "Billboards breathe electric prayers,\n"
            "Somewhere, a quiet signal blinks,\n"
            "waiting for someone to notice.",

            f"{base}.\n"
            "Chrome and moonlight share a secret,\n"
            "Cables curl like restless thoughts,\n"
            "A heartbeat hides in every screen,\n"
            "and the future watches from the corner.",
        ]
        return random.choice(templates)

    if mode == "funny":
        setups = [
            f"{base.capitalize()}, which is exactly what happens",
            f"{base.capitalize()} — honestly, this is what you get",
            f"{base.capitalize()}. This is the natural result",
        ]
        punches = [
            " when you let an AI design the future after three energy drinks.",
            " when nobody reads the terms and conditions but clicks accept anyway.",
            " when the Wi-Fi is strong but the life choices are weak.",
            " when someone says, 'How bad could it be?' and the universe takes it personally.",
        ]
        endings = [
            " Somewhere, a server is facepalming in 4K.",
            " An AI just added this to its ‘are you sure?’ folder.",
            " The cloud is laughing, but your storage is full.",
        ]
        return random.choice(setups) + random.choice(punches) + " " + random.choice(endings)

    # Default: plain caption
    return caption



# Load the captioning model at startup
logger.info("Initializing image captioning model...")
try:
    get_caption_model()  # This loads the model at startup
    logger.info("Image captioning model loaded successfully")
except Exception as e:
    logger.error(f"Error loading captioning model: {str(e)}")
    logger.warning("App will continue but caption generation may fail")

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle image upload and display with caption generation and audio"""
    uploaded_image = None
    caption = None
    audio_path = None
    mode = 'caption'
    emotion = 'calm'
    
    if request.method == 'POST':
        # Check if file was uploaded
        mode = request.form.get('mode', 'caption')
        emotion = request.form.get('emotion', 'calm')
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        # Validate and save file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image = filepath
            
            # Generate caption for the uploaded image
            try:
                logger.info(f"Generating caption for image: {filepath}")
                base_caption = generate_caption(filepath)
                logger.info(f"Base caption: {base_caption}")

                caption = apply_mode_to_caption(base_caption, mode)
                logger.info(f"Final text for mode '{mode}': {caption}")

                
                # Generate audio from caption
                if caption and caption != "Unable to generate caption":
                    try:
                        logger.info("Generating audio from caption...")
                        audio_path = text_to_audio(
                            caption, 
                            app.config['AUDIO_FOLDER'], 
                            filename,
                            emotion
                        )
                        if audio_path:
                            logger.info(f"Audio generated successfully: {audio_path}")
                        else:
                            logger.warning("Failed to generate audio")
                    except Exception as e:
                        logger.error(f"Error generating audio: {str(e)}")
                        flash(f'Error generating audio: {str(e)}')
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                flash(f'Error generating caption: {str(e)}')
                # Continue to display image even if caption generation fails
                caption = "Unable to generate caption"
        else:
            flash('Invalid file type. Please upload an image (png, jpg, jpeg, gif, webp)')
            return redirect(request.url)
    
    return render_template('index.html', uploaded_image=uploaded_image, caption=caption, audio_path=audio_path, mode=mode, emotion=emotion)

if __name__ == '__main__':
    # Create uploads and audio directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8000)