import cv2
import torch # type: ignore
import numpy as np
import pytesseract # type: ignore
from PIL import Image # type: ignore
import easyocr # type: ignore
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRProcessor:
    """
    Main OCR class for handling text extraction from images.
    Combines multiple OCR methods for better accuracy.
    """
    def __init__(self):
        # Initialize EasyOCR for text detection
        self.reader = easyocr.Reader(['en'], gpu=True if torch.cuda.is_available() else False)
        
        # Initialize TrOCR for more accurate text recognition
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        if torch.cuda.is_available():
            self.trocr_model.to('cuda')
    
    def get_text_from_region(self, img):
        """Extract all text from an image region using multiple OCR methods"""
        # EasyOCR
        easy_results = self.reader.readtext(img)
        easy_text = " ".join([text for _, text, _ in easy_results]) if easy_results else ""
        
        # Tesseract OCR (as backup)
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            tess_text = pytesseract.image_to_string(thresh)
        except:
            tess_text = ""
        
        # TrOCR (for higher accuracy)
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
            pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            generated_ids = self.trocr_model.generate(pixel_values)
            tr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except:
            tr_text = ""
        
        # Combine all results
        combined_text = f"{easy_text} {tess_text} {tr_text}"
        return combined_text