"""Image preprocessing module to clean up and optimize images for OCR."""

from dataclasses import dataclass
from typing import List
import numpy as np
import cv2
import PIL.Image


@dataclass
class PreprocessorConfig:
    """Configuration class for image preprocessing settings."""
    grayscale:          bool = True
    denoise:            bool = True
    denoise_strength:   int  = 10        # OpenCV h parameter for fastNlMeansDenoising
    deskew:             bool = True
    deskew_threshold:   float = 0.5      # degrees; skip if rotation < threshold
    binarize:           bool = True
    binarize_method:    str  = "otsu"    # "otsu" | "adaptive" | "sauvola"
    contrast_enhance:   bool = True
    contrast_clip:      float = 2.0      # CLAHE clip limit
    upscale_if_small:   bool = True
    upscale_min_width:  int  = 1500      # px; upscale if image width < this
    remove_borders:     bool = False     # crop black/white scan borders


HANDWRITING_PRESET = PreprocessorConfig(
    grayscale=True,
    denoise=True,
    denoise_strength=15,
    deskew=True,
    binarize=True,
    binarize_method="adaptive",   # better for uneven lighting in handwriting
    contrast_enhance=True,
    contrast_clip=3.0,
    upscale_if_small=True,
    remove_borders=False,
)

PRINTED_PRESET = PreprocessorConfig(
    grayscale=True,
    denoise=True,
    denoise_strength=10,
    deskew=True,
    binarize=True,
    binarize_method="otsu",
    contrast_enhance=False,
    upscale_if_small=True,
    remove_borders=False,
)


class ImagePreprocessor:
    """Applies preprocessing pipeline steps to PIL Images for optimal Tesseract OCR performance."""

    def __init__(self, config: PreprocessorConfig) -> None:
        """Initializes the preprocessor with a specific configuration.

        Args:
            config: PreprocessorConfig instance.
        """
        self.config = config

    def _sauvola_threshold(self, img: np.ndarray, window_size: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
        """Applies Sauvola local thresholding to a grayscale image.

        Sauvola threshold is: T = m * (1 + k * (s / R - 1))
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local mean
        mean = cv2.blur(img, (window_size, window_size))
        
        # Calculate local mean of square
        mean_sq = cv2.blur(img.astype(np.float32)**2, (window_size, window_size))
        
        # Calculate local standard deviation
        variance = mean_sq - mean.astype(np.float32)**2
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)
        
        # Sauvola threshold
        threshold = mean * (1.0 + k * (std / R - 1.0))
        
        # Binarize
        binary = np.where(img >= threshold, 255, 0).astype(np.uint8)
        return binary

    def _grayscale(self, img: np.ndarray) -> np.ndarray:
        """Converts BGR image to grayscale (single-channel)."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _contrast_enhance(self, img: np.ndarray) -> np.ndarray:
        """Applies CLAHE contrast enhancement."""
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.config.contrast_clip, tileGridSize=(8, 8))
            return clahe.apply(img)
        else:
            # For BGR images, apply CLAHE on the L channel of LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.config.contrast_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Applies fastNlMeansDenoising to reduce noise."""
        h = self.config.denoise_strength
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
        else:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Aligns horizontal text blocks by rotating the image based on minAreaRect angle."""
        gray = self._grayscale(img)
        # Use Otsu binarization to extract characters/lines for angle detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Get coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return img
        
        # minAreaRect needs coordinates in (x, y) format
        coords_xy = coords[:, ::-1]
        rect = cv2.minAreaRect(coords_xy)
        angle = rect[-1]
        
        # Normalize angle to [-45, 45] degrees
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
            
        if abs(angle) < self.config.deskew_threshold:
            return img
            
        # Rotate image around its center
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Determine background color for empty areas after rotation
        border_val = 255 if len(img.shape) == 2 else (255, 255, 255)
        
        rotated = cv2.warpAffine(
            img, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=border_val
        )
        return rotated

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Converts image to binary black-and-white format using configured method."""
        gray = self._grayscale(img)
        
        if self.config.binarize_method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif self.config.binarize_method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
            )
        elif self.config.binarize_method == "sauvola":
            return self._sauvola_threshold(gray)
        else:
            return gray

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscales image using bicubic interpolation if width is below threshold."""
        h, w = img.shape[:2]
        if w < self.config.upscale_min_width:
            scale_factor = self.config.upscale_min_width / w
            new_w = self.config.upscale_min_width
            new_h = int(h * scale_factor)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """Removes black or white scanned borders by finding main document content boundary."""
        gray = self._grayscale(img)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
            
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Skip noise or full page border contours
            if w < 10 or h < 10:
                continue
            if w > img.shape[1] * 0.99 and h > img.shape[0] * 0.99:
                continue
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            
        if x_max > x_min and y_max > y_min:
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img.shape[1], x_max + padding)
            y_max = min(img.shape[0], y_max + padding)
            return img[y_min:y_max, x_min:x_max]
            
        return img

    def process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Runs the complete preprocessing pipeline on a PIL Image.

        Pipeline steps are applied in an optimal sequence:
        1. Grayscale conversion
        2. Denoising
        3. Contrast Enhancement
        4. Deskewing (rotation)
        5. Binarization
        6. Upscaling
        7. Border removal
        """
        # Convert PIL to Numpy BGR
        np_img = np.array(image.convert("RGB"))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # 1. Grayscale
        if self.config.grayscale or self.config.binarize:
            np_img = self._grayscale(np_img)

        # 2. Denoise
        if self.config.denoise:
            np_img = self._denoise(np_img)

        # 3. Contrast Enhance
        if self.config.contrast_enhance:
            np_img = self._contrast_enhance(np_img)

        # 4. Deskew
        if self.config.deskew:
            np_img = self._deskew(np_img)

        # 5. Binarize
        if self.config.binarize:
            np_img = self._binarize(np_img)

        # 6. Upscale
        if self.config.upscale_if_small:
            np_img = self._upscale(np_img)

        # 7. Remove Borders
        if self.config.remove_borders:
            np_img = self._remove_borders(np_img)

        # Convert back to PIL Image
        if len(np_img.shape) == 2:
            return PIL.Image.fromarray(np_img, mode="L")
        else:
            np_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            return PIL.Image.fromarray(np_rgb, mode="RGB")

    def process_batch(self, images: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
        """Processes a batch of PIL Images sequentially.

        Args:
            images: List of PIL Image objects.

        Returns:
            List of processed PIL Image objects.
        """
        return [self.process(img) for img in images]
