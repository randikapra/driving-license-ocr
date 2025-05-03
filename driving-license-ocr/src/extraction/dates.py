## Improved OCR is dividing two sections. one for date extraction one for vehicle class extraction 

import re
import cv2
import numpy as np
import pytesseract # type: ignore
from src.extraction.ocr import OCRProcessor
from src.preprocessing.enhancement import enhance_cell_image

class DateExtractor:
    """
    Class for extracting and validating dates from Sri Lankan driving licenses.
    """
    def __init__(self, ocr_processor=None):
        # Initialize OCR processor if not provided
        self.ocr_processor = ocr_processor if ocr_processor else OCRProcessor()
        
        # Enhanced date patterns for Sri Lankan format
        self.date_patterns = [
            r'\d{2}[.]\d{2}[.]\d{4}',    # DD.MM.YYYY
            r'\d{2}[-]\d{2}[-]\d{4}',     # DD-MM-YYYY
            r'\d{2}[/]\d{2}[/]\d{4}',     # DD/MM/YYYY
            r'\d{2}\s*\d{2}\s*\d{4}',     # DD MM YYYY (with or without spaces)
            r'\d{2}[.]\d{2}[.]\d{2}',     # DD.MM.YY
            r'\d{1,2}[.]\d{1,2}[.]\d{4}', # D.M.YYYY
            r'\d{1,2}[.]\d{1,2}[.]\d{2}', # D.M.YY
            r'\d{1,2}\s*\d{1,2}\s*\d{4}', # D M YYYY
            r'\d{4}'                       # Just year as last resort
        ]
    
    def extract_dates_from_row(self, table_img, row_y, row_h):
        """Extract start and end dates from a row with improved detection"""
        # Get the region of the row with padding
        padding = 15  # Increased padding
        row_start = max(0, row_y - padding)
        row_end = min(table_img.shape[0], row_y + row_h + padding)
        row_img = table_img[row_start:row_end, :]
        
        # Calculate column positions specifically for Sri Lankan license format
        width = row_img.shape[1]
        
        # Adaptive column positions based on image width
        # Try multiple column positions since the exact position can vary
        start_col_ratios = [0.60, 0.65, 0.70]  # Try multiple positions for start date
        end_col_ratios = [0.78, 0.82, 0.85]    # Try multiple positions for end date
        
        start_date = None
        end_date = None
        
        # Try multiple positions for start date
        for ratio in start_col_ratios:
            if start_date:
                break
                
            start_col_x = int(width * ratio) - 40  # With margin
            col_width = int(width * 0.25)  # Width large enough to capture full date
            
            # Extract column image
            start_date_col = row_img[:, max(0, start_col_x):min(width, start_col_x+col_width)]
            
            # Enhance date column image
            enhanced_start_col = enhance_cell_image(start_date_col)
            
            # Extract date using multiple OCR methods
            start_date = self.extract_date_from_image(enhanced_start_col)
        
        # Try multiple positions for end date
        for ratio in end_col_ratios:
            if end_date:
                break
                
            end_col_x = int(width * ratio) - 40  # With margin
            col_width = int(width * 0.25)  # Width large enough to capture full date
            
            # Extract column image
            end_date_col = row_img[:, max(0, end_col_x):min(width, end_col_x+col_width)]
            
            # Enhance date column image
            enhanced_end_col = enhance_cell_image(end_date_col)
            
            # Extract date using multiple OCR methods
            end_date = self.extract_date_from_image(enhanced_end_col)
        
        # Try a broader region if dates not found
        if not start_date:
            # Try entire right half of the row
            right_half = row_img[:, int(width*0.5):]
            enhanced_right = enhance_cell_image(right_half)
            
            # Extract text from the right half
            right_text = self.ocr_processor.get_text_from_region(enhanced_right)
            
            # Try to find dates in the text
            all_dates = []
            for pattern in self.date_patterns:
                matches = re.findall(pattern, right_text)
                all_dates.extend(matches)
                
            # Sort dates by position in text
            if len(all_dates) >= 2:
                start_date = self.normalize_date_format(all_dates[0])
                end_date = self.normalize_date_format(all_dates[1])
            elif len(all_dates) == 1:
                # If only one date found, use it as start date
                start_date = self.normalize_date_format(all_dates[0])
        
        # Also try Tesseract OCR as a fallback
        if not start_date or not end_date:
            try:
                # Extract the right portion of the row which should contain dates
                right_half = row_img[:, int(width*0.5):]
                # Apply preprocessing optimized for dates
                gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY) if len(right_half.shape) == 3 else right_half
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Use Tesseract with specific configuration for digits and punctuation
                config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./-'
                tess_text = pytesseract.image_to_string(thresh, config=config)
                
                # Extract dates from Tesseract results
                all_dates = []
                for pattern in self.date_patterns:
                    matches = re.findall(pattern, tess_text)
                    all_dates.extend(matches)
                
                if len(all_dates) >= 2:
                    if not start_date:
                        start_date = self.normalize_date_format(all_dates[0])
                    if not end_date:
                        end_date = self.normalize_date_format(all_dates[1])
                elif len(all_dates) == 1:
                    if not start_date:
                        start_date = self.normalize_date_format(all_dates[0])
                    elif not end_date:
                        end_date = self.normalize_date_format(all_dates[0])
            except Exception as e:
                # If Tesseract fails, continue with what we have
                pass
        
        # Last resort - try to find any 4-digit year in the row
        if not start_date and not end_date:
            row_text = self.ocr_processor.get_text_from_region(row_img)
            year_matches = re.findall(r'(19|20)\d{2}', row_text)
            
            if year_matches:
                # Use the year to construct a date
                year = year_matches[0]
                start_date = f"01.01.{year}"
                if len(year_matches) > 1:
                    end_date = f"01.01.{year_matches[1]}"
        
        return start_date or "Not Found", end_date or "Not Found"
    
    def extract_date_from_image(self, img):
        """Extract date from image using OCR and regex patterns"""
        # Get text from image using OCR
        text = self.ocr_processor.get_text_from_region(img)
        
        # Search for date patterns in the extracted text
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the first matched date
                return self.normalize_date_format(matches[0])
        
        return None
    
    def normalize_date_format(self, date_str):
        """Normalize date format to DD.MM.YYYY"""
        # Implementation of date normalization logic
        # This would convert various date formats to a standard format
        return date_str  # Placeholder - actual implementation would normalize formats