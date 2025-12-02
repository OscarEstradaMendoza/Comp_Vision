"""
Image Captioning Model Module using Hugging Face BLIP

This module provides a wrapper around the Hugging Face BLIP (Bootstrapped Language-Image
Pretraining) model for generating natural language captions from images.

Key Features:
- Loads and manages the BLIP image captioning model
- Optimized for M1 Mac with MPS (Metal Performance Shaders) GPU acceleration
- Automatic device detection (MPS -> CPU fallback)
- Singleton pattern for efficient model reuse
- Error handling and logging

Model Information:
- Model: Salesforce/blip-image-captioning-base
- Size: ~990MB (downloaded automatically on first use)
- Device: MPS (Apple Silicon GPU) or CPU
- Framework: PyTorch with Transformers library

Dependencies:
- torch: PyTorch deep learning framework
- transformers: Hugging Face transformers library
- PIL (Pillow): Image processing
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioningModel:
    """
    Wrapper class for the BLIP image captioning model.
    
    This class manages the lifecycle of the BLIP model, including:
    - Device detection and selection (MPS for M1 Mac, CPU fallback)
    - Model and processor loading
    - Image caption generation
    
    The model is loaded once during initialization and reused for all
    subsequent caption generation requests, which is much more efficient
    than loading it for each request.
    
    Attributes:
        device (torch.device): The device where the model is loaded (MPS or CPU)
        processor (BlipProcessor): The BLIP processor for image preprocessing
        model (BlipForConditionalGeneration): The BLIP model for caption generation
    """
    
    def __init__(self):
        """
        Initialize the ImageCaptioningModel instance.
        
        This constructor:
        1. Detects and sets the best available device (MPS or CPU)
        2. Loads the BLIP processor and model
        3. Moves the model to the selected device
        
        Raises:
            Exception: If model loading fails (e.g., network issues, disk space)
        """
        # Initialize attributes (will be set by helper methods)
        self.device = None
        self.processor = None
        self.model = None
        
        # Initialize device detection and model loading
        self._initialize_device()
        self._load_model()
    
    def _initialize_device(self):
        """
        Detect and set the best available device for model execution.
        
        This method checks for MPS (Metal Performance Shaders) availability,
        which provides GPU acceleration on Apple Silicon Macs (M1, M2, etc.).
        If MPS is not available, it falls back to CPU.
        
        Device Priority:
        1. MPS (Apple Silicon GPU) - Fastest, uses GPU acceleration
        2. CPU - Fallback option, slower but works on all systems
        
        Note:
            MPS is only available on macOS with Apple Silicon processors.
            On other systems, this will automatically use CPU.
        """
        # Check if MPS (Metal Performance Shaders) is available
        # MPS provides GPU acceleration on Apple Silicon Macs
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU) for acceleration")
        else:
            # Fallback to CPU if MPS is not available
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
    
    def _load_model(self):
        """
        Load the BLIP model and processor from Hugging Face.
        
        This method:
        1. Downloads the model from Hugging Face (if not already cached)
        2. Loads the processor for image preprocessing
        3. Loads the model for caption generation
        4. Moves the model to the selected device (MPS or CPU)
        
        Model Details:
        - Name: Salesforce/blip-image-captioning-base
        - Type: Vision-Language model for image captioning
        - First download: ~990MB (cached locally after first use)
        
        Raises:
            Exception: If model download or loading fails
                Common causes:
                - Network connectivity issues
                - Insufficient disk space
                - Corrupted model cache
        """
        try:
            logger.info("Loading BLIP image captioning model...")
            model_name = "Salesforce/blip-image-captioning-base"
            
            # Load the processor
            # The processor handles image preprocessing (resizing, normalization, etc.)
            # and text tokenization for the model
            self.processor = BlipProcessor.from_pretrained(model_name)
            logger.info("Processor loaded successfully")
            
            # Load the model and move it to the selected device
            # The model is the actual neural network that generates captions
            # Moving to device (MPS/CPU) is necessary for inference
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a natural language caption for an image.
        
        This method processes an image through the BLIP model to generate
        a descriptive caption. The process involves:
        1. Loading and preprocessing the image
        2. Running inference through the model
        3. Decoding the model output to text
        
        Args:
            image_path (str): Path to the image file
                Supported formats: JPEG, PNG, GIF, WebP, etc. (any PIL-supported format)
        
        Returns:
            str: Generated caption describing the image
                Example: "a group of people standing next to each other"
        
        Raises:
            Exception: If image loading or caption generation fails
                Common causes:
                - Invalid image file
                - Corrupted image data
                - Model inference errors
        
        Note:
            The model generates captions in English. The caption is typically
            a short, descriptive sentence about the main subject of the image.
        """
        try:
            # Step 1: Load and convert image to RGB format
            # PIL's convert('RGB') ensures consistent color channel format
            # This is important because images may have different formats (RGBA, grayscale, etc.)
            raw_image = Image.open(image_path).convert('RGB')
            
            # Step 2: Preprocess the image using the BLIP processor
            # This includes:
            # - Resizing to model's expected input size
            # - Normalizing pixel values
            # - Converting to tensor format
            # - Moving to the correct device (MPS or CPU)
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            
            # Step 3: Generate caption using the model
            # torch.no_grad() disables gradient computation for efficiency
            # (we're doing inference, not training)
            # max_length=50: Maximum caption length in tokens
            # num_beams=3: Beam search width for better quality captions
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=3)
            
            # Step 4: Decode the model output to readable text
            # The model outputs token IDs, which need to be converted to words
            # skip_special_tokens=True removes special tokens like [PAD], [CLS], etc.
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise


# ============================================================================
# SINGLETON PATTERN IMPLEMENTATION
# ============================================================================
# Global variable to store the model instance
# This ensures the model is loaded only once and reused across all requests
_caption_model = None


def get_caption_model() -> ImageCaptioningModel:
    """
    Get or initialize the global caption model instance (Singleton pattern).
    
    This function implements the Singleton pattern to ensure the model is loaded
    only once, even if called multiple times. This is crucial for performance
    because model loading is expensive (takes several seconds and uses memory).
    
    The first call loads the model, subsequent calls return the same instance.
    
    Returns:
        ImageCaptioningModel: The global model instance
    
    Example:
        >>> model1 = get_caption_model()  # Loads model
        >>> model2 = get_caption_model()   # Returns same instance (no reload)
        >>> model1 is model2
        True
    """
    global _caption_model
    
    # Lazy initialization: only create if it doesn't exist
    if _caption_model is None:
        _caption_model = ImageCaptioningModel()
    
    return _caption_model


def generate_caption(image_path: str) -> str:
    """
    Convenience function to generate a caption for an image.
    
    This is a simple wrapper around the ImageCaptioningModel class that:
    1. Gets the global model instance (loading if necessary)
    2. Calls the generate_caption method
    3. Returns the result
    
    This function provides a simple interface for the Flask application
    without needing to manage the model instance directly.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Generated caption describing the image
    
    Example:
        >>> caption = generate_caption("photo.jpg")
        >>> print(caption)
        "a cat sitting on a windowsill"
    """
    # Get the global model instance (Singleton pattern)
    model = get_caption_model()
    
    # Generate and return the caption
    return model.generate_caption(image_path)
