"""
Image Captioning Model using Hugging Face BLIP
Optimized for M1 Mac with MPS (Metal Performance Shaders) support
"""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioningModel:
    """Wrapper class for BLIP image captioning model"""
    
    def __init__(self):
        """Initialize the model and processor"""
        self.device = None
        self.processor = None
        self.model = None
        self._initialize_device()
        self._load_model()
    
    def _initialize_device(self):
        """Detect and set the best available device (MPS for M1 Mac, fallback to CPU)"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU) for acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
    
    def _load_model(self):
        """Load the BLIP model and processor"""
        try:
            logger.info("Loading BLIP image captioning model...")
            model_name = "Salesforce/blip-image-captioning-base"
            
            # Load processor
            self.processor = BlipProcessor.from_pretrained(model_name)
            logger.info("Processor loaded successfully")
            
            # Load model and move to device
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a caption for the given image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption string
        """
        try:
            # Load and convert image to RGB
            raw_image = Image.open(image_path).convert('RGB')
            
            # Process the image
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=3)
            
            # Decode the caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise


# Global model instance (loaded once at startup)
_caption_model = None


def get_caption_model():
    """Get or initialize the global caption model instance"""
    global _caption_model
    if _caption_model is None:
        _caption_model = ImageCaptioningModel()
    return _caption_model


def generate_caption(image_path: str) -> str:
    """
    Convenience function to generate a caption for an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Generated caption string
    """
    model = get_caption_model()
    return model.generate_caption(image_path)

