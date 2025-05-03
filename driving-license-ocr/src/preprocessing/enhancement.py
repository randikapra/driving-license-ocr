import cv2
import numpy as np

def enhance_image_for_ocr(img):
    """
    Apply image enhancement techniques optimized for OCR
    """
    """
    Enhance the entire image for better OCR results.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        Enhanced image for OCR processing
    """
    # Handle empty images
    if img is None or img.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # 2. Noise reduction
    denoised = cv2.fastNlMeansDenoising(clahe_img, None, 10, 7, 21)
    
    # 3. Adaptive thresholding optimized for documents
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    
    # 4. Dilation to slightly thicken text
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 5. Opening to remove small noise
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    
    return opening



def enhance_cell_image(cell_img):
    """
    Enhance cell image specifically for better OCR results in table cells.
    
    Args:
        cell_img: Image of a table cell (numpy array)
        
    Returns:
        Enhanced cell image for OCR processing
    """
    # Handle empty images
    if cell_img is None or cell_img.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
        
    # Convert to grayscale if needed
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
        
    # Resize for better OCR if too small
    if gray.shape[0] < 30 or gray.shape[1] < 100:
        scale_factor = 2
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Invert if needed (white text on black background)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
        
    # Morphological operations to clean up noise
    kernel = np.ones((2, 1), np.uint8)  # More horizontal to preserve digits
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return enhanced