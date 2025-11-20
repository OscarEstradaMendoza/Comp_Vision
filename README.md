# Computer Vision Image Captioning & Text-to-Speech

A cyberpunk-themed Flask web application that generates captions for uploaded images using Hugging Face's BLIP model and converts captions to audio using Google Text-to-Speech.

## Features

- 🖼️ Image upload and display
- 🤖 AI-powered image captioning using BLIP model
- 🔊 Text-to-speech audio generation from captions
- 🎨 Futuristic cyberpunk-themed UI
- ⚡ Optimized for M1 Mac with MPS (Metal Performance Shaders) support
- 📝 Multiple caption modes: Caption, Story, Poetic, Funny
- 🎭 Multiple emotion styles: Calm, Energetic, Sad, Robotic, Whisper

## Requirements

- Python 3.8+
- M1 Mac (or compatible system with 16GB+ RAM)
- Internet connection (for model downloads and text-to-speech)
- ffmpeg (for emotion-based audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OscarEstradaMendoza/Comp_Vision.git
cd Comp_Vision
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg (for audio emotion processing):
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Upload an image, select a mode (Caption, Story, Poetic, or Funny) and an emotion (Calm, Energetic, Sad, Robotic, or Whisper), then click UPLOAD to generate a caption and hear it spoken aloud!

## Project Structure

```
CVProject/
├── app.py                 # Main Flask application
├── caption_model.py       # BLIP image captioning model wrapper
├── audio_generator.py     # Text-to-speech audio generation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Cyberpunk-themed frontend
└── static/
    ├── uploads/          # Uploaded images
    └── audio/            # Generated audio files
```

## Technologies Used

- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - BLIP image captioning model
- **gTTS (Google Text-to-Speech)** - Text-to-audio conversion
- **Pillow** - Image processing
- **ffmpeg** - Audio processing for emotion effects

## Model Information

- **Image Captioning Model**: `Salesforce/blip-image-captioning-base`
- **Device**: MPS (Metal Performance Shaders) for M1 Mac acceleration
- **Model Size**: ~990MB (downloaded automatically on first run)

## Notes

- First run will download the BLIP model (~990MB), which may take a few minutes
- Requires internet connection for text-to-speech generation
- Browser may block auto-play of audio; use the play button if needed
- Different modes transform the caption in creative ways (poetic, story, funny)
- Different emotions apply audio effects to the generated speech

## License

MIT License
