## --version 5.2.5.-- ##
import cv2
import numpy as np
import pandas as pd
import re
import easyocr # type: ignore
import torch # type: ignore
from pathlib import Path
from PIL import Image # type: ignore
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt # type: ignore
import pytesseract # type: ignore

class ImprovedDrivingLicenseExtractor:
    def __init__(self):
        # Initialize EasyOCR for text detection
        self.reader = easyocr.Reader(['en'], gpu=True if torch.cuda.is_available() else False)
        
        # Initialize TrOCR for more accurate text recognition
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        if torch.cuda.is_available():
            self.trocr_model.to('cuda')
        
        # Define vehicle class codes with correct formats
        self.vehicle_classes = ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J']
        
        # Enhanced class corrections with more variations
        self.class_corrections = {
            'A1': ['A1', 'Al', 'AI', 'A!', 'A|', 'Ai'],
            'A': ['A'],
            'B1': ['B1', 'Bl', 'BI', 'B!', 'B|', 'Bi'],
            'B': ['B', '8'],
            'C1': ['C1', 'Cl', 'CI', 'C!', 'C|', 'Ci'],
            'C': ['C'],
            'CE': ['CE', 'Ce', 'cE', 'ce'],
            'D1': ['D1', 'Dl', 'DI', 'D!', 'D|', 'Di'],
            'D': ['D', 'O'],
            'DE': ['DE', 'De', 'dE', 'de'],
            'G1': ['G1', 'Gl', 'GI', 'G!', 'G|', 'Gi'],
            'G': ['G', '6'],
            'J': ['J', 'j']
        }
        
        # Build reverse lookup dictionary
        self.misread_to_correct = {}
        for correct, misreads in self.class_corrections.items():
            for misread in misreads:
                self.misread_to_correct[misread] = correct
        
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
        
    def correct_vehicle_class(self, detected_class):
        """Correct common OCR misreads for vehicle classes"""
        # Handle None or empty string
        if detected_class is None or not detected_class.strip():
            return None
            
        # Convert to uppercase for consistency
        detected_class = detected_class.upper().strip()
        
        # Direct lookup in misread dictionary
        if detected_class in self.misread_to_correct:
            return self.misread_to_correct[detected_class]
        
        # Check if it's already a valid class
        if detected_class in self.vehicle_classes:
            return detected_class
            
        # Try similarity matching for less common errors
        best_match = None
        max_similarity = 0
        
        # Enhanced similarity check that handles shorter strings
        for valid_class in self.vehicle_classes:
            # For single character vehicle classes like 'A', 'B', etc.
            if len(valid_class) == 1 and len(detected_class) == 1:
                if detected_class == valid_class:
                    return valid_class
            
            # For longer classes, use character-by-character comparison
            elif len(detected_class) >= len(valid_class) * 0.5:  # Allow for partial matches
                # Count matching characters in sequence
                matches = 0
                for i in range(min(len(detected_class), len(valid_class))):
                    if i < len(detected_class) and i < len(valid_class) and detected_class[i] == valid_class[i]:
                        matches += 1
                
                similarity = matches / len(valid_class)
                if similarity > max_similarity and similarity > 0.5:  # At least 50% match
                    max_similarity = similarity
                    best_match = valid_class
        
        return best_match if best_match else detected_class

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
            enhanced_start_col = self.enhance_cell_image(start_date_col)
            
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
            enhanced_end_col = self.enhance_cell_image(end_date_col)
            
            # Extract date using multiple OCR methods
            end_date = self.extract_date_from_image(enhanced_end_col)
        
        # Try a broader region if dates not found
        if not start_date:
            # Try entire right half of the row
            right_half = row_img[:, int(width*0.5):]
            enhanced_right = self.enhance_cell_image(right_half)
            
            # Extract text from the right half
            right_text = self.get_text_from_region(enhanced_right)
            
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
            row_text = self.get_text_from_region(row_img)
            year_matches = re.findall(r'(19|20)\d{2}', row_text)
            
            if year_matches:
                # Use the year to construct a date
                year = year_matches[0]
                start_date = f"01.01.{year}"
                if len(year_matches) > 1:
                    end_date = f"01.01.{year_matches[1]}"
        
        return start_date or "Not Found", end_date or "Not Found"
    
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
    
    def enhance_cell_image(self, cell_img):
        """Enhance cell image for better OCR results"""
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

    def extract_date_from_image(self, img):
        """Extract date from image using multiple methods"""
        # Try TrOCR first
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            generated_ids = self.trocr_model.generate(pixel_values)
            tr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            tr_text = ""
            
        # Try EasyOCR as backup
        try:
            easy_results = self.reader.readtext(img)
            easy_text = " ".join([text for _, text, _ in easy_results]) if easy_results else ""
        except Exception as e:
            easy_text = ""
        
        # Try Tesseract OCR with optimizations for dates
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                
            # Apply threshold optimized for date digits
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Use Tesseract with specific config for dates
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789./-'
            tess_text = pytesseract.image_to_string(thresh, config=config)
        except Exception as e:
            tess_text = ""
        
        # Combine all results
        all_text = f"{tr_text} {easy_text} {tess_text}"
        
        # Look for date patterns
        for pattern in self.date_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                return self.normalize_date_format(matches[0])
        
        # Try digit reconstruction for cases where dots are missed
        digits = re.findall(r'\d{2}', all_text)
        if len(digits) >= 3:
            # Try to construct a date from consecutive 2-digit groups
            for i in range(len(digits)-2):
                potential_date = f"{digits[i]}.{digits[i+1]}.{digits[i+2]}"
                if self.is_valid_date(potential_date):
                    return potential_date
        
        # Check for standalone years (4 digits)
        years = re.findall(r'(19|20)\d{2}', all_text)
        if years:
            # Just return the year if that's all we can find
            return years[0]
                    
        return None
    
    def normalize_date_format(self, date_str):
        """Normalize different date formats to DD.MM.YYYY"""
        # Replace separators with consistent format
        normalized = re.sub(r'[/-]', '.', date_str)
        
        # Remove spaces if any
        normalized = normalized.replace(' ', '')
        
        # Handle different formats
        if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', normalized):
            # Ensure DD.MM.YYYY format with leading zeros
            parts = normalized.split('.')
            day = parts[0].zfill(2)
            month = parts[1].zfill(2)
            year = parts[2]
            return f"{day}.{month}.{year}"
        elif re.match(r'\d{1,2}\.\d{1,2}\.\d{2}', normalized):
            # Convert YY to YYYY
            parts = normalized.split('.')
            day = parts[0].zfill(2)
            month = parts[1].zfill(2)
            year = parts[2]
            # Assume 20YY for years less than 50, 19YY otherwise
            if int(year) < 50:
                year = f"20{year.zfill(2)}"
            else:
                year = f"19{year.zfill(2)}"
            return f"{day}.{month}.{year}"
        elif re.match(r'\d{4}', normalized):
            # Just a year - return as 01.01.YYYY
            return f"01.01.{normalized}"
            
        return date_str
    
    def is_valid_date(self, date_str):
        """Check if a string represents a valid date"""
        try:
            # Try to parse as DD.MM.YYYY
            parts = date_str.split('.')
            if len(parts) != 3:
                return False
                
            day, month, year = map(int, parts)
            
            # Basic validation
            if not (1 <= day <= 31 and 1 <= month <= 12 and 1990 <= year <= 2040):
                return False
                
            return True
        except:
            return False
            
    def detect_vehicle_rows(self, table_img):
        """
        Detect rows with vehicle class information with improved accuracy
        Returns: List of (y, height, vehicle_class) tuples
        """
        # Handle empty images
        if table_img is None or table_img.size == 0:
            return []
            
        # Convert to grayscale if not already
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY) if len(table_img.shape) == 3 else table_img
        
        # Enhance the image for better OCR
        enhanced = self.enhance_image_for_detection(gray)
        
        # Calculate expected row height based on image dimensions
        row_height_estimate = int(table_img.shape[0] / 14)  # Assuming about 13-14 rows in license
        
        # Use multiple OCR methods to detect potential vehicle classes
        vehicle_code_positions = []
        
        # Method 1: Use EasyOCR to detect text
        try:
            results = self.reader.readtext(enhanced)
            
            for (bbox, text, prob) in results:
                text = text.strip().upper()
                
                # Check if it's a potential vehicle class
                if len(text) <= 3:  # Vehicle classes are short (A1, B, CE, etc.)
                    # Correct common OCR errors in vehicle classes
                    corrected_class = self.correct_vehicle_class(text)
                    
                    # Check if it's a valid vehicle class after correction
                    if corrected_class in self.vehicle_classes:
                        top_left = tuple(map(int, bbox[0]))
                        bottom_right = tuple(map(int, bbox[2]))
                        y = top_left[1]
                        height = bottom_right[1] - top_left[1]
                        
                        # Add a position tuple with (y position, box height, class code)
                        vehicle_code_positions.append((y, height, corrected_class))
        except Exception as e:
            print(f"EasyOCR detection error: {e}")
        
        # Method 2: Use grid-based approach
        if len(vehicle_code_positions) < 5:
            height, width = table_img.shape[:2]
            
            # Calculate positions where vehicle classes usually appear (left side of the table)
            class_column_ratios = [0.15, 0.18, 0.2]  # Try multiple ratios
            
            for ratio in class_column_ratios:
                class_column_x = int(width * ratio)
                
                # Create synthetic rows based on estimated row height
                for i in range(13):  # For each potential vehicle class row
                    row_y = int(height * 0.12) + i * row_height_estimate  # Start from 12% down
                    
                    # Extract region where vehicle class should be
                    class_region_width = int(width * 0.15)
                    class_region = table_img[row_y:row_y+row_height_estimate, 
                                           max(0, class_column_x-10):class_column_x+class_region_width]
                    
                    if class_region.size == 0:
                        continue
                        
                    # Try to detect vehicle class in this region
                    enhanced_region = self.enhance_cell_image(class_region)
                    
                    # Try multiple OCR methods
                    class_text = ""
                    
                    # EasyOCR
                    try:
                        region_results = self.reader.readtext(enhanced_region)
                        for _, text, _ in region_results:
                            class_text += text + " "
                    except:
                        pass
                    
                    # Tesseract
                    try:
                        tess_text = pytesseract.image_to_string(enhanced_region, config='--psm 10')
                        class_text += tess_text + " "
                    except:
                        pass
                    
                    # Process the combined text
                    for text in class_text.split():
                        text = text.strip().upper()
                        corrected_class = self.correct_vehicle_class(text)
                        
                        if corrected_class in self.vehicle_classes:
                            vehicle_code_positions.append((row_y, row_height_estimate, corrected_class))
                            break
        
        # Method 3: Target search for specific vehicle classes
        if len(vehicle_code_positions) < 5:
            for vehicle_class in self.vehicle_classes:
                # Convert the class to image template for template matching
                # (This is a simplified approach - could be enhanced with actual templates)
                for letter_scale in np.linspace(0.8, 1.5, 5):
                    # Create image from text
                    text_img = np.zeros((50, 100), dtype=np.uint8)
                    cv2.putText(text_img, vehicle_class, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                              letter_scale, 255, 2)
                    
                    # Try to match this template in the image
                    try:
                        # Use edge detection to make matching more robust
                        edges = cv2.Canny(enhanced, 100, 200)
                        template_edges = cv2.Canny(text_img, 100, 200)
                        
                        res = cv2.matchTemplate(edges, template_edges, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.4:  # Reasonable threshold for template matching
                            y = max_loc[1]
                            vehicle_code_positions.append((y, row_height_estimate, vehicle_class))
                    except:
                        continue
        
        # Method 4: Forced grid detection as last resort
        if len(vehicle_code_positions) < 5:
            height, width = table_img.shape[:2]
            header_height = int(height * 0.12)  # Skip header area
            content_height = height - header_height
            
            # Assuming at least 5 rows for vehicle classes
            row_height = content_height / 13
            
            # For each potential class, check both sides of the image
            for class_idx, vehicle_class in enumerate(self.vehicle_classes):
                row_y = header_height + class_idx * row_height
                vehicle_code_positions.append((int(row_y), int(row_height), vehicle_class))
        
        # Sort by y-coordinate to get rows in order
        vehicle_code_positions.sort(key=lambda x: x[0])
        
        # Remove duplicates that are too close to each other (same row)
        deduped_positions = []
        if vehicle_code_positions:
            deduped_positions.append(vehicle_code_positions[0])
            last_y = vehicle_code_positions[0][0]
            last_class = vehicle_code_positions[0][2]
            
            for y, h, vc in vehicle_code_positions[1:]:
                # Skip if too close to previous position or same class
                if abs(y - last_y) <= h/2 or vc == last_class:
                    continue
                    
                deduped_positions.append((y, h, vc))
                last_y = y
                last_class = vc
        
        return deduped_positions
    
    def enhance_image_for_detection(self, img):
        """Enhance image for better text detection"""
        # Apply bilateral filter to preserve edges while reducing noise
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
                                    
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morph

    def enhance_image_for_text_extraction(self, img):
        """Enhance image specifically for text extraction"""
        # Handle None or empty image
        if img is None or img.size == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(cl, 9, 75, 75)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        # Convert back to BGR for OCR compatibility
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return enhanced

    def process_image(self, image_path):
        """
        Main function to process an image and extract license details
        using the preprocessed image.
        """
        # Step 1: Use the provided preprocessing function
        preprocessing_results = preprocess_license_image(image_path)
        
        if preprocessing_results is None or 'table' not in preprocessing_results:
            print("Preprocessing failed or table region not found")
            return pd.DataFrame(columns=['Vehicle Class', 'Start Date', 'Expiry Date'])
        
        # Step 2: Get the table region from preprocessing results
        table_img = preprocessing_results['table']
        
        # Skip additional preprocessing as it's already done by preprocess_license_image
        # Instead, focus on enhancing specifically for text extraction
        enhanced_table = self.enhance_image_for_text_extraction(table_img)
        
        # Step 3: Detect rows with vehicle class information using improved methods
        vehicle_rows = self.detect_vehicle_rows(enhanced_table)
        
        # Step 4: Extract dates for each vehicle class row
        license_data = []
        
        # If we have vehicle rows, process each one
        if vehicle_rows:
            # For each detected vehicle class
            for (row_y, row_h, vehicle_class) in vehicle_rows:
                # Extract dates from this row
                start_date, expiry_date = self.extract_dates_from_row(enhanced_table, row_y, row_h)
                
                # Add to results
                license_data.append({
                    'Vehicle Class': vehicle_class,
                    'Start Date': start_date,
                    'Expiry Date': expiry_date
                })
        else:
            # Fallback: Try grid-based approach if no rows detected
            print("No vehicle class rows detected, trying grid-based approach")
            license_data = self.extract_license_data_grid_based(enhanced_table)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(license_data)
        
        # Filter out any rows where start date and expiry date are both "Not Found"
        # Only if we have other valid rows
        if len(result_df) > 1:
            result_df = result_df[~((result_df['Start Date'] == 'Not Found') & 
                                   (result_df['Expiry Date'] == 'Not Found'))]
        
        return result_df
    
    def extract_license_data_grid_based(self, table_img):
        """
        Fallback method using a grid-based approach to extract license data
        when regular row detection fails.
        """
        license_data = []
        height, width = table_img.shape[:2]
        
        # Calculate typical row height based on license format
        # Typically, Sri Lankan licenses have around 13 potential rows for vehicle classes
        header_height = int(height * 0.12)  # Skip header area
        usable_height = height - header_height
        row_height = usable_height / 13  # Approximate height for each row
        
        # Check for each vehicle class
        for i, vehicle_class in enumerate(self.vehicle_classes):
            # Calculate row position
            row_y = int(header_height + i * row_height)
            
            if row_y >= height - row_height:
                continue  # Skip if row would be outside image
            
            # Try to verify if this vehicle class is actually present
            # Extract class column region
            class_col_x = int(width * 0.15)  # Left side of table where classes usually appear
            class_region_width = int(width * 0.15)
            class_region = table_img[row_y:int(row_y+row_height), 
                                   max(0, class_col_x-15):class_col_x+class_region_width]
            
            # Skip if empty
            if class_region.size == 0:
                continue
                
            # Enhance class region for better OCR
            enhanced_region = self.enhance_cell_image(class_region)
            
            # Detect text in class region
            easy_results = self.reader.readtext(enhanced_region)
            detected_text = " ".join([text for _, text, _ in easy_results]) if easy_results else ""
            
            # If no text detected, try TrOCR
            if not detected_text:
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(enhanced_region, cv2.COLOR_BGR2RGB))
                    pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
                    if torch.cuda.is_available():
                        pixel_values = pixel_values.to('cuda')
                    
                    generated_ids = self.trocr_model.generate(pixel_values)
                    detected_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                except:
                    detected_text = ""
            
            # Check if current vehicle class is detected
            found_match = False
            for text_segment in detected_text.split():
                corrected = self.correct_vehicle_class(text_segment)
                if corrected == vehicle_class:
                    found_match = True
                    break
            
            # If we didn't find this class, continue to next class
            if not found_match and len(detected_text) > 0:
                continue
            
            # Extract dates from the row
            start_date, expiry_date = self.extract_dates_from_row(table_img, row_y, int(row_height))
            
            # Only add rows where we have some data
            if start_date != "Not Found" or expiry_date != "Not Found":
                license_data.append({
                    'Vehicle Class': vehicle_class,
                    'Start Date': start_date,
                    'Expiry Date': expiry_date
                })
        
        return license_data
    
    def extract_dates_from_row(self, table_img, row_y, row_h):
        """Extract start and end dates from a row with improved detection"""
        # Get the region of the row with padding
        padding = 15  # Increased padding
        row_start = max(0, row_y - padding)
        row_end = min(table_img.shape[0], row_y + row_h + padding)
        row_img = table_img[row_start:row_end, :]
        
        # Calculate column positions specifically for Sri Lankan license format
        width = row_img.shape[1]
        
        # Improved adaptive column positions
        # Different license formats might have columns in different positions
        start_date_column_options = [
            (0.40, 0.65),  # Format 1
            (0.45, 0.70),  # Format 2
            (0.50, 0.70),  # Format 3
            (0.55, 0.75)   # Format 4
        ]
        
        end_date_column_options = [
            (0.68, 0.90),  # Format 1
            (0.70, 0.95),  # Format 2
            (0.75, 0.95),  # Format 3
            (0.80, 1.00)   # Format 4
        ]
        
        start_date = None
        end_date = None
        
        # Try all column position options for start date
        for start_col_start, start_col_end in start_date_column_options:
            if start_date:
                break
                
            start_col_x = int(width * start_col_start)
            col_width = int(width * (start_col_end - start_col_start))
            
            # Extract column image
            start_date_col = row_img[:, max(0, start_col_x):min(width, start_col_x+col_width)]
            
            if start_date_col.size == 0:
                continue
                
            # Apply multiple image enhancement techniques
            enhanced_start_cols = [
                self.enhance_cell_image(start_date_col),  # Standard enhancement
                cv2.cvtColor(cv2.threshold(cv2.cvtColor(start_date_col, cv2.COLOR_BGR2GRAY), 
                                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR),  # OTSU threshold
                cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(start_date_col, cv2.COLOR_BGR2GRAY), 
                                                 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), 
                           cv2.COLOR_GRAY2BGR)  # Adaptive threshold
            ]
            
            # Try each enhanced image for OCR
            for enhanced_col in enhanced_start_cols:
                # Try multiple OCR methods
                extracted_date = self.extract_date_from_image(enhanced_col)
                if extracted_date:
                    start_date = extracted_date
                    break
        
        # Try all column position options for end date
        for end_col_start, end_col_end in end_date_column_options:
            if end_date:
                break
                
            end_col_x = int(width * end_col_start)
            col_width = int(width * (end_col_end - end_col_start))
            
            # Extract column image
            end_date_col = row_img[:, max(0, end_col_x):min(width, end_col_x+col_width)]
            
            if end_date_col.size == 0:
                continue
                
            # Apply multiple image enhancement techniques
            enhanced_end_cols = [
                self.enhance_cell_image(end_date_col),  # Standard enhancement
                cv2.cvtColor(cv2.threshold(cv2.cvtColor(end_date_col, cv2.COLOR_BGR2GRAY), 
                                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR),  # OTSU threshold
                cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(end_date_col, cv2.COLOR_BGR2GRAY), 
                                                 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), 
                           cv2.COLOR_GRAY2BGR)  # Adaptive threshold
            ]
            
            # Try each enhanced image for OCR
            for enhanced_col in enhanced_end_cols:
                # Try multiple OCR methods
                extracted_date = self.extract_date_from_image(enhanced_col)
                if extracted_date:
                    end_date = extracted_date
                    break
        
        # If dates still not found, try full-row analysis with multiple techniques
        if not start_date or not end_date:
            # Try the entire row
            enhanced_row = self.enhance_cell_image(row_img)
            
            # Method 1: Use EasyOCR
            results = self.reader.readtext(enhanced_row)
            full_text = " ".join([text for _, text, _ in results]) if results else ""
            
            # Method 2: Use Tesseract with specific config
            try:
                gray_row = cv2.cvtColor(enhanced_row, cv2.COLOR_BGR2GRAY)
                config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./-'
                tess_text = pytesseract.image_to_string(gray_row, config=config)
                full_text += " " + tess_text
            except:
                pass
            
            # Method 3: Use TrOCR
            try:
                pil_img = Image.fromarray(cv2.cvtColor(enhanced_row, cv2.COLOR_BGR2RGB))
                pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
                if torch.cuda.is_available():
                    pixel_values = pixel_values.to('cuda')
                
                generated_ids = self.trocr_model.generate(pixel_values)
                tr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                full_text += " " + tr_text
            except:
                pass
            
            # Extract all dates from the combined text
            all_dates = []
            for pattern in self.date_patterns:
                matches = re.findall(pattern, full_text)
                all_dates.extend(matches)
            
            # If we found at least two dates, use them
            if len(all_dates) >= 2:
                all_dates = [self.normalize_date_format(date) for date in all_dates]
                # Sort by position in text (earlier = start date, later = end date)
                if not start_date:
                    start_date = all_dates[0]
                if not end_date:
                    end_date = all_dates[1] if len(all_dates) > 1 else None
            elif len(all_dates) == 1:
                # If only one date found, use it for whichever is missing
                normalized_date = self.normalize_date_format(all_dates[0])
                if not start_date:
                    start_date = normalized_date
                elif not end_date:
                    end_date = normalized_date
        
        # Final fallback for dates
        if not start_date or not end_date:
            # Extract all 4-digit sequences that could be years
            year_pattern = r'(19|20)\d{2}'
            year_matches = re.findall(year_pattern, full_text if 'full_text' in locals() else "")
            
            if year_matches:
                # Use years to construct dates
                if len(year_matches) >= 2:
                    if not start_date:
                        start_date = f"01.01.{year_matches[0]}"
                    if not end_date:
                        end_date = f"01.01.{year_matches[1]}"
                elif len(year_matches) == 1:
                    if not start_date:
                        start_date = f"01.01.{year_matches[0]}"
                    # For end date, assume validity period of 8 years (common for Sri Lankan licenses)
                    if not end_date:
                        try:
                            end_year = int(year_matches[0]) + 8
                            end_date = f"01.01.{end_year}"
                        except:
                            pass
        
        return start_date or "Not Found", end_date or "Not Found"
    
    def extract_date_from_image(self, img):
        """Extract date from image using multiple methods with improved accuracy"""
        if img is None or img.size == 0:
            return None
            
        # Use multiple OCR methods to increase chances of detecting dates
        extracted_text = ""
        
        # 1. TrOCR - usually most accurate for printed text
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
            pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            generated_ids = self.trocr_model.generate(pixel_values)
            tr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted_text += tr_text + " "
        except Exception as e:
            pass
        
        # 2. EasyOCR - good for detecting text regions
        try:
            easy_results = self.reader.readtext(img)
            easy_text = " ".join([text for _, text, _ in easy_results]) if easy_results else ""
            extracted_text += easy_text + " "
        except Exception as e:
            pass
        
        # 3. Tesseract with specific configuration for dates
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply multiple thresholding methods and try each
            thresh_methods = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            ]
            
            for thresh in thresh_methods:
                # Configurations optimized for date extraction
                configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789./-',  # Single line, optimized for dates
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./-',  # Assume single uniform block of text
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789./-'   # Treat as single word
                ]
                
                for config in configs:
                    tess_text = pytesseract.image_to_string(thresh, config=config)
                    extracted_text += tess_text + " "
        except Exception as e:
            pass
        
        # Look for date patterns in extracted text
        for pattern in self.date_patterns:
            matches = re.findall(pattern, extracted_text)
            if matches:
                return self.normalize_date_format(matches[0])
        
        # Try digit sequence reconstruction for fragmented dates
        digits = re.findall(r'\d+', extracted_text)
        if len(digits) >= 3:
            # Try to construct a date from consecutive digit groups
            for i in range(len(digits)-2):
                day = digits[i].zfill(2) if len(digits[i]) <= 2 else digits[i][:2]
                month = digits[i+1].zfill(2) if len(digits[i+1]) <= 2 else digits[i+1][:2]
                
                # Year could be 2 or 4 digits
                if len(digits[i+2]) == 4:
                    year = digits[i+2]
                elif len(digits[i+2]) == 2:
                    year_val = int(digits[i+2])
                    year = f"20{digits[i+2]}" if year_val < 50 else f"19{digits[i+2]}"
                else:
                    continue
                
                potential_date = f"{day}.{month}.{year}"
                if self.is_valid_date(potential_date):
                    return potential_date
        
        # If still nothing, look for isolated years
        years = re.findall(r'(19|20)\d{2}', extracted_text)
        if years:
            # Rather than just return the year, construct a full date
            return f"01.01.{years[0]}"
        
        return None
    
    def detect_vehicle_rows(self, table_img):
        """
        Enhanced method to detect rows with vehicle class information
        Returns list of (y, height, vehicle_class) tuples
        """
        if table_img is None or table_img.size == 0:
            return []
        
        height, width = table_img.shape[:2]
        detected_rows = []
        
        # Method 1: Direct OCR detection of vehicle classes
        # This focuses on finding the actual vehicle class codes
        try:
            # First, optimize the image for text detection
            if len(table_img.shape) == 3:
                gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = table_img.copy()
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply threshold to further enhance text
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use EasyOCR to find text in the image
            results = self.reader.readtext(binary)
            
            # Filter results to find vehicle class codes
            for (bbox, text, prob) in results:
                if prob < 0.2:  # Skip low confidence detections
                    continue
                    
                text = text.strip().upper()
                
                # Correct common OCR errors in vehicle classes
                corrected_class = self.correct_vehicle_class(text)
                
                # Check if it's a valid vehicle class after correction
                if corrected_class in self.vehicle_classes:
                    # Calculate row position and height
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    y = top_left[1]
                    h = bottom_right[1] - top_left[1]
                    
                    # Add to detected rows
                    detected_rows.append((y, h, corrected_class))
        except Exception as e:
            print(f"Error in direct OCR detection: {e}")
        
        # Method 2: Examine left side of table for class codes
        # Sri Lankan driving licenses typically have vehicle classes on the left side
        try:
            # Calculate where the vehicle class column should be
            class_col_width = int(width * 0.25)  # Column width based on typical license format
            
            # Try multiple horizontal positions for the class column
            for col_ratio in [0.05, 0.1, 0.15, 0.2]:
                col_x = int(width * col_ratio)
                
                # Extract the class column region
                class_col = table_img[:, col_x:col_x+class_col_width]
                
                # Enhanced preprocessing specific for class column
                if len(class_col.shape) == 3:
                    gray_col = cv2.cvtColor(class_col, cv2.COLOR_BGR2GRAY)
                else:
                    gray_col = class_col.copy()
                
                # Apply adaptive threshold to better distinguish text
                thresh_col = cv2.adaptiveThreshold(gray_col, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
                
                # Try to find all vehicle classes in this column
                for vehicle_class in self.vehicle_classes:
                    # Skip classes we've already found
                    if any(vc == vehicle_class for _, _, vc in detected_rows):
                        continue
                    
                    # Direct text search using OCR
                    results = self.reader.readtext(thresh_col)
                    
                    # Check each result for this vehicle class
                    for (bbox, text, prob) in results:
                        text = text.strip().upper()
                        
                        # Try different normalization approaches
                        corrected = self.correct_vehicle_class(text)
                        
                        if corrected == vehicle_class:
                            # Calculate row position and height
                            top_left = tuple(map(int, bbox[0]))
                            bottom_right = tuple(map(int, bbox[2]))
                            y = top_left[1]
                            h = bottom_right[1] - top_left[1]
                            
                            detected_rows.append((y, h, vehicle_class))
        except Exception as e:
            print(f"Error in class column search: {e}")
        
        # Method 3: Grid-based approach for more comprehensive detection
        if len(detected_rows) < 5:  # If few rows detected, try grid approach
            try:
                # Calculate estimated row height based on table dimensions
                est_row_height = int(height / 14)  # Assuming around 13-14 rows including header
                
                # Skip header area
                header_height = int(height * 0.12)
                
                # Horizontal position where classes usually appear
                class_x = int(width * 0.15)
                class_width = int(width * 0.15)
                
                # Check each potential row position
                for i in range(13):  # Check up to 13 potential rows for vehicle classes
                    # Calculate row position
                    row_y = header_height + i * est_row_height
                    
                    # Skip if outside image bounds
                    if row_y >= height - est_row_height:
                        continue
                    
                    # Extract class region
                    class_region = table_img[row_y:row_y+est_row_height, class_x:class_x+class_width]
                    
                    if class_region.size == 0:
                        continue
                    
                    # Try each potential vehicle class at this position
                    for vehicle_class in self.vehicle_classes:
                        # Skip classes we've already found
                        if any(vc == vehicle_class for _, _, vc in detected_rows):
                            continue
                            
                        # Enhance region for better detection
                        enhanced_region = self.enhance_cell_image(class_region)
                        
                        # Use OCR to detect text
                        easy_results = self.reader.readtext(enhanced_region)
                        region_text = " ".join([text for _, text, _ in easy_results]) if easy_results else ""
                        
                        # Try all possible variation checks
                        for variation in self.class_corrections.get(vehicle_class, []):
                            if variation.upper() in region_text.upper():
                                detected_rows.append((row_y, est_row_height, vehicle_class))
                                break
            except Exception as e:
                print(f"Error in grid-based approach: {e}")
        
        # Sort rows by vertical position and remove duplicates
        if detected_rows:
            detected_rows.sort(key=lambda x: x[0])
            
            # Remove duplicate classes and rows that are too close to each other
            unique_rows = []
            last_y = -1000
            last_class = ""
            
            for y, h, vehicle_class in detected_rows:
                # Skip if too close to previous row or duplicate class
                if (abs(y - last_y) <= h*0.7 and last_class == vehicle_class) or vehicle_class == last_class:
                    continue
                
                unique_rows.append((y, h, vehicle_class))
                last_y = y
                last_class = vehicle_class
            
            return unique_rows
        
        # Fallback if all methods failed: create rows based on expected layout
        if not detected_rows:
            print("Using fallback row detection")
            # Create synthetic rows for common vehicle classes
            common_classes = ['B', 'C1', 'C', 'CE', 'G1']
            header_height = int(height * 0.12)
            est_row_height = int((height - header_height) / 13)
            
            for i, vehicle_class in enumerate(common_classes):
                row_y = header_height + i * est_row_height
                detected_rows.append((row_y, est_row_height, vehicle_class))
        
        return detected_rows

    def complete_processing(self, image_path):
        """
        Complete pipeline to process an image and create a comprehensive report.
        Includes visualization options for debugging.
        """
        # Process the image 
        result_df = self.process_image(image_path)
        
        # Visualize results
        print("Extracted License Information:")
        print(result_df)
        
        # Return the DataFrame for further processing
        return result_df

    def visualize_detected_info(self, image_path):
        """
        Visualize the detected information on the image
        Parameters:
        - image_path: Path to the license image
        """
        # Preprocess the image
        preprocessed = preprocess_license_image(image_path)
        if not preprocessed:
            return
         
        # Get the table image
        table_img = preprocessed["table_roi"].copy()
         
        # # Display the image
        # plt.figure(figsize=(15, 10))
        # # plt.imshow(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))
        # plt.imshow(table_img)
        # plt.title(f"Detected Information in {Path(image_path).name}")
        # plt.axis('off')
        # plt.show()
         
        # Detect rows with vehicle class information
        vehicle_rows = self.detect_vehicle_rows(table_img)
         
        # Draw rectangles around detected rows
        for y, height, vehicle_class in vehicle_rows:
            cv2.rectangle(table_img, (0, y), (table_img.shape[1], y+height), (0, 255, 0), 2)
            cv2.putText(table_img, vehicle_class, (10, y+height//2), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                 
            # Extract dates from the row
            start_date, end_date = self.extract_dates_from_row(table_img, y, height)
                 
            # Display dates
            if start_date != "Not Found":
                cv2.putText(table_img, f"Start: {start_date}", (table_img.shape[1]//3, y+height//2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                 
            if end_date != "Not Found":
                cv2.putText(table_img, f"End: {end_date}", (2*table_img.shape[1]//3, y+height//2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
         
        # Display the image
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))
        # plt.imshow(table_img)
        plt.title(f"Detected Information in {Path(image_path).name}")
        plt.axis('off')
        plt.show()


