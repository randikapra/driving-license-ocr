import cv2
import numpy as np
import math
from collections import Counter
import pytesseract # type: ignore
import re

def determine_orientation(img):
    """
    Determine correct orientation using a combined approach of:
    1. Table structure detection
    2. OCR confidence at different orientations
    3. License-specific pattern recognition
    4. Enhanced text direction detection
    """
    # Try all four major orientations
    orientations = [0, 90, 180, 270]
    orientation_scores = []
    rotated_images = []
    
    # For each orientation, calculate a combined score
    for angle in orientations:
        # Rotate image
        rotated = rotate_image(img, angle)
        rotated_images.append(rotated)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        
        # 1. Score based on table structure detection
        table_score = score_table_structure(gray)
        
        # 2. Score based on OCR confidence
        ocr_score = score_ocr_confidence(gray)
        
        # 3. Score based on license-specific patterns
        pattern_score = score_license_patterns(gray)
        
        # 4. NEW: Score based on text direction and header/footer detection
        text_direction_score = score_text_direction(gray)
        
        # Calculate weighted total score with improved weights
        # We prioritize license patterns and text direction as they're most reliable
        total_score = (table_score * 0.15) + (ocr_score * 0.40) + \
                      (pattern_score * 0.025) + (text_direction_score * 0.35)
        
        # Debug print to track orientation scores
        print(f"Angle {angle}° - Table: {table_score:.2f}, OCR: {ocr_score:.2f}, " + 
              f"Pattern: {pattern_score:.2f}, Text Dir: {text_direction_score:.2f}, Total: {total_score:.2f}")
        
        orientation_scores.append(total_score)
    
    # Find the best orientation
    best_idx = np.argmax(orientation_scores)
    best_angle = orientations[best_idx]
    best_img = rotated_images[best_idx]
    
    print(f"Selected best orientation: {best_angle}°")
    
    # Apply fine rotation correction for small angles
    if best_angle == 0 or best_angle == 180:  # Only apply minor correction to 0 and 180 degrees
        corrected_img, minor_angle = correct_minor_rotation(best_img)
        total_angle = (best_angle + minor_angle) % 360
    else:
        corrected_img = best_img
        total_angle = best_angle
    
    # IMPROVED: More reliable upside-down detection
    # If we're at 0° or 180°, double-check the orientation using a more robust method
    if total_angle == 0 or total_angle == 180:
        if is_image_upside_down(corrected_img):
            print("License appears to be upside down, flipping image...")
            corrected_img = rotate_image(corrected_img, 180)
            total_angle = (total_angle + 180) % 360
    
    return corrected_img, total_angle

def score_text_direction(gray_img):
    """
    NEW FUNCTION: Score based on text direction and header/footer positioning
    For Sri Lankan licenses, specific header text should be at the top
    and department information at the bottom when correctly oriented
    """
    h, w = gray_img.shape
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract top, middle, and bottom regions
    top_region = thresh[:h//4, :]
    middle_region = thresh[h//4:3*h//4, :]
    bottom_region = thresh[3*h//4:, :]
    
    # Run OCR on each region
    top_text = pytesseract.image_to_string(top_region)
    middle_text = pytesseract.image_to_string(middle_region)
    bottom_text = pytesseract.image_to_string(bottom_region)
    
    score = 0
    
    # Check for header keywords in top region
    header_keywords = ['Driving', 'License', 'Licence', 'Republic', 'Sri', 'Lanka']
    header_matches = sum(1 for keyword in header_keywords if keyword.lower() in top_text.lower())
    header_score = min(1.0, header_matches / 3)
    
    # Check for department keywords in bottom region
    dept_keywords = ['Department', 'Motor', 'Traffic', 'DMT']
    dept_matches = sum(1 for keyword in dept_keywords if keyword.lower() in bottom_text.lower())
    dept_score = min(1.0, dept_matches / 2)
    
    # Check for typical table content in middle region
    table_keywords = ['Class', 'Category', 'Date', 'Valid', 'Issue', 'Expiry']
    table_matches = sum(1 for keyword in table_keywords if keyword.lower() in middle_text.lower())
    table_score = min(1.0, table_matches / 3)
    
    # Check if bottom text contains more potential category codes than top text
    category_pattern = r'\b[A-G][1]?\b'  # Matches A, A1, B, B1, etc.
    top_categories = len(re.findall(category_pattern, top_text))
    bottom_categories = len(re.findall(category_pattern, bottom_text))
    category_position_score = 1.0 if top_categories <= bottom_categories else 0.0
    
    # Calculate combined score
    score = (header_score * 0.4) + (dept_score * 0.3) + \
            (table_score * 0.2) + (category_position_score * 0.1)
    
    return score * 100

def is_image_upside_down(img):
    """
    IMPROVED FUNCTION: More reliably detect if license is upside down
    Combines multiple checks for better accuracy
    """
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h, w = thresh.shape
    
    # Run OCR on top quarter and bottom quarter
    top_quarter = thresh[:h//4, :]
    bottom_quarter = thresh[3*h//4:, :]
    
    top_text = pytesseract.image_to_string(top_quarter).lower()
    bottom_text = pytesseract.image_to_string(bottom_quarter).lower()
    
    # 1. Check for department keywords
    dept_keywords = ['department', 'motor', 'traffic', 'sri', 'lanka', 'dmt']
    dept_in_top = sum(1 for keyword in dept_keywords if keyword in top_text)
    dept_in_bottom = sum(1 for keyword in dept_keywords if keyword in bottom_text)
    
    # 2. Check for header keywords
    header_keywords = ['driving', 'license', 'licence', 'republic']
    header_in_top = sum(1 for keyword in header_keywords if keyword in top_text)
    header_in_bottom = sum(1 for keyword in header_keywords if keyword in bottom_text)
    
    # 3. Look for date patterns (assume dates are more likely in bottom half when upright)
    date_pattern = r'\d{2}[./-]\d{2}[./-]\d{4}'
    dates_in_top = len(re.findall(date_pattern, top_text))
    dates_in_bottom = len(re.findall(date_pattern, bottom_text))
    
    # 4. Check for signature line in bottom (common in upright licenses)
    signature_keywords = ['sign', 'signature', 'holder']
    sig_in_top = sum(1 for keyword in signature_keywords if keyword in top_text)
    sig_in_bottom = sum(1 for keyword in signature_keywords if keyword in bottom_text)
    
    # Calculate evidence scores for upside-down orientation
    upside_down_evidence = 0
    
    # Department text should be in bottom, not top
    if dept_in_top > dept_in_bottom:
        upside_down_evidence += 2
    
    # Header should be in top, not bottom
    if header_in_bottom > header_in_top:
        upside_down_evidence += 2
    
    # Dates are typically more common in bottom half
    if dates_in_top > dates_in_bottom and dates_in_top > 0:
        upside_down_evidence += 1
    
    # Signature line should be in bottom
    if sig_in_top > sig_in_bottom and sig_in_top > 0:
        upside_down_evidence += 1
    
    # Make final determination
    # Higher threshold for flipping to avoid unnecessary flips
    return upside_down_evidence >= 3

def is_text_upside_down(img):
    """
    LEGACY FUNCTION: Original check if the text in the image appears to be upside down
    Kept for backwards compatibility
    """
    # Just delegate to the improved function
    return is_image_upside_down(img)

def score_table_structure(gray_img):
    """
    Score image based on detected table structure
    Higher score = more likely to be correct orientation
    """
    # Detect edges
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100, 
                          minLineLength=gray_img.shape[1]//4, maxLineGap=20)
    
    if lines is None:
        return 0
    
    # Count horizontal and vertical lines
    h_lines = 0
    v_lines = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line angle
        if x2 - x1 == 0:  # Vertical line
            v_lines += 1
        else:
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 20 or angle > 160:  # Horizontal
                h_lines += 1
            elif 70 < angle < 110:  # Vertical
                v_lines += 1
    
    # Calculate score - Sri Lankan driving licenses typically have
    # a grid structure with both horizontal and vertical lines
    # We want to detect a proper grid/table structure
    
    total_lines = h_lines + v_lines
    if total_lines == 0:
        return 0
    
    # We expect more horizontal than vertical lines in the license table
    # and we want a good balance of both horizontal and vertical lines
    h_ratio = h_lines / total_lines if total_lines > 0 else 0
    v_ratio = v_lines / total_lines if total_lines > 0 else 0
    
    # Ideal is to have both h_lines and v_lines but with more h_lines
    # This score will be highest when we have a good number of both types,
    # but with horizontal lines being more numerous (typical for license tables)
    balance_score = (h_ratio * 0.6 + v_ratio * 0.4) * min(1.0, total_lines / 10)
    
    # Additional score for detecting a grid-like pattern
    # Check for intersections of horizontal and vertical lines
    grid_score = min(1.0, (h_lines * v_lines) / 100) if h_lines > 0 and v_lines > 0 else 0
    
    return (balance_score * 0.7 + grid_score * 0.3) * 100

def score_ocr_confidence(gray_img):
    """
    Score image based on OCR confidence
    Also looks for text in expected positions
    """
    try:
        # Apply adaptive thresholding for better OCR
        thresh = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Run OCR with detailed output
        ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config='--psm 6')
        
        # Calculate basic confidence
        confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
        if not confidences:
            return 0
        avg_confidence = sum(confidences) / len(confidences)
        
        # Count recognizable words (length > 1)
        word_count = sum(1 for word in ocr_data['text'] if len(word) > 1)
        
        # Calculate text positions score
        # License has text in expected regions - header, footer, and table area
        h, w = gray_img.shape
        top_text = 0
        middle_text = 0
        bottom_text = 0
        
        # Analyze where text is found
        for i, text in enumerate(ocr_data['text']):
            if not text or len(text) <= 1:
                continue
                
            # Get center y-position of text
            y_pos = ocr_data['top'][i] + ocr_data['height'][i] / 2
            
            # Categorize position
            if y_pos < h * 0.25:  # Top 25%
                top_text += 1
            elif y_pos > h * 0.75:  # Bottom 25%
                bottom_text += 1
            else:  # Middle 50%
                middle_text += 1
        
        # Sri Lankan licenses typically have text at top (title), 
        # middle (table), and bottom (department info)
        position_score = min(1.0, (top_text + bottom_text + middle_text) / 10)
        
        # Calculate final score: combination of confidence, word count, and position
        base_score = (avg_confidence / 100.0) * min(1.0, word_count / 20.0)
        return (base_score * 0.7 + position_score * 0.3) * 100
        
    except Exception as e:
        print(f"Error in OCR confidence calculation: {e}")
        return 0

def score_license_patterns(gray_img):
    """
    IMPROVED: Score image based on license-specific patterns with better weighting
    """
    try:
        # Apply thresholding for better text detection
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR for pattern detection
        text = pytesseract.image_to_string(thresh)
        lower_text = text.lower()  # Convert to lowercase for case-insensitive matching
        
        score = 0
        
        # 1. Check for license categories - very specific to driving licenses
        categories = ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J']
        category_matches = sum(1 for cat in categories if cat in text)
        category_score = min(1.0, category_matches / 4)  # Normalize to max of 1.0
        
        # 2. Check for date patterns (DD.MM.YYYY or DD/MM/YYYY)
        date_pattern = r'\d{2}[./-]\d{2}[./-]\d{4}'
        dates_found = len(re.findall(date_pattern, text))
        date_score = min(1.0, dates_found / 3)  # Normalize to max of 1.0
        
        # 3. Check for restriction text (common in licenses)
        restriction_keywords = ['restriction', 'condition', 'valid', 'from', 'to', 'expiry']
        restriction_matches = sum(1 for keyword in restriction_keywords if keyword.lower() in lower_text)
        restriction_score = min(1.0, restriction_matches / 3)
        
        # 4. Look for license number pattern (alphanumeric)
        license_num_pattern = r'[A-Z]\d{6,}'  # License numbers often start with letter followed by digits
        license_match = 1.0 if re.search(license_num_pattern, text) else 0.0
        
        # Calculate weighted score - stronger weighting for most reliable indicators
        weighted_score = (category_score * 0.4 + 
                         date_score * 0.3 + 
                         restriction_score * 0.2 + 
                         license_match * 0.1) * 100
        print(f'category_score:{category_score}, date_score:{date_score}, restriction_score:{restriction_score}, license_match:{license_match}')                 
        weighted_score = 0
        return weighted_score
        
    except Exception as e:
        print(f"Error in pattern recognition: {e}")
        return 0

def rotate_image(img, angle):
    """Rotate image by specified angle"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, M, (w, h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_minor_rotation(img):
    """
    Correct minor rotation issues using Hough transform
    to detect dominant line angles
    """
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                          minLineLength=gray.shape[1]//5, maxLineGap=10)
    
    if lines is None:
        return img, 0
    
    # Calculate angles of lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip very short lines
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 50:
            continue
            
        # Calculate angle, avoiding division by zero
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            
        # Normalize angle to range [-45, 45]
        while angle < -45:
            angle += 180
        while angle > 45:
            angle -= 180
            
        angles.append(angle)
    
    if not angles:
        return img, 0
        
    # Find dominant angle
    # Group similar angles to get more robust estimate
    angle_counts = Counter(round(a) for a in angles)
    if not angle_counts:
        return img, 0
        
    dominant_angle = angle_counts.most_common(1)[0][0]
    
    # Skip very small corrections
    if abs(dominant_angle) < 0.5:
        return img, 0
        
    # Apply rotation to correct the minor angle
    corrected = rotate_image(img, dominant_angle)
    return corrected, dominant_angle