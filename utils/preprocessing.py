import cv2
import numpy as np
import logging
from typing import Tuple

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config['preprocessing']
        logging.info("Initialized ImagePreprocessor with config")
        
    def estimate_illumination(self, image: np.ndarray) -> float:
        """Estimate overall image illumination."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def adaptive_clahe(self, image: np.ndarray, illumination: float) -> np.ndarray:
        """Apply adaptive CLAHE based on illumination level."""
        clip_limit = self.config['clahe']['clip_limit']
        # Adjust clip limit based on illumination
        if illumination < 100:  # Dark image
            clip_limit *= 1.5
        elif illumination > 200:  # Bright image
            clip_limit *= 0.7
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tuple(self.config['clahe']['tile_grid_size'])
        )
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            # Estimate illumination
            illumination = self.estimate_illumination(image)
            
            # Resize
            image = cv2.resize(image, tuple(self.config['resize_dims']))
            
            # Apply adaptive CLAHE
            image = self.adaptive_clahe(image, illumination)
            
            # Apply bilateral filter for edge-preserving smoothing
            image = cv2.bilateralFilter(
                image,
                d=self.config['bilateral']['diameter'],
                sigmaColor=self.config['bilateral']['sigma_color'],
                sigmaSpace=self.config['bilateral']['sigma_space']
            )
            
            # Apply Gaussian blur
            image = cv2.GaussianBlur(
                image,
                tuple(self.config['gaussian_blur']['kernel_size']),
                self.config['gaussian_blur']['sigma']
            )
            
            # Apply unsharp masking
            gaussian = cv2.GaussianBlur(
                image,
                tuple(self.config['unsharp_masking']['kernel_size']),
                self.config['unsharp_masking']['sigma']
            )
            image = cv2.addWeighted(
                image, 1 + self.config['unsharp_masking']['amount'],
                gaussian, -self.config['unsharp_masking']['amount'], 0
            )
            
            return image
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise
