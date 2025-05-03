import cv2
import numpy as np

def detect_license_table(enhanced_img, original_img):
    """
    Detect and extract the table region containing vehicle classes and dates
    """
    """
    Detect the license table structure in the image.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        Tuple containing:
        - table_img: Cropped image containing only the table
        - rows: List of (y, height) for each detected row
        - success: Boolean indicating if table was successfully detected
    """
    # Create copy of original for visualization
    vis_img = original_img.copy() if len(original_img.shape) > 2 else cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # Convert enhanced image to binary if not already
    if len(enhanced_img.shape) > 2:
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced_img.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Prepare variables for ROI
    table_roi = None
    x, y, w, h = 0, 0, binary.shape[1], binary.shape[0]
    
    # Look for table-like structures
    # Typically the table of vehicle classes is a large rectangular region
    for contour in contours[:10]:  # Check top 10 contours
        area = cv2.contourArea(contour)
        
        # Skip small contours
        if area < binary.shape[0] * binary.shape[1] * 0.05:
            continue
            
        # Get bounding rectangle
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w_rect) / h_rect
        
        # Check if it's a reasonable rectangle for a license table
        # Tables are typically wider than tall
        if 0.7 < aspect_ratio < 3.0:
            # Check if it covers a significant portion of the image
            relative_area = area / (binary.shape[0] * binary.shape[1])
            if relative_area > 0.2:
                x, y, w, h = x_rect, y_rect, w_rect, h_rect
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                break
    
    # Extract the table ROI
    table_roi = gray[y:y+h, x:x+w]
    
    return vis_img, table_roi


import cv2
import numpy as np

def detect_license_table_ocr(img):
    """
    Detect the license table structure in the image.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        Tuple containing:
        - table_img: Cropped image containing only the table
        - rows: List of (y, height) for each detected row
        - success: Boolean indicating if table was successfully detected
    """
    # Make a copy of the input image
    original = img.copy()
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect horizontal lines (rows)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines (columns)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine horizontal and vertical lines
    table_structure = cv2.add(horizontal_lines, vertical_lines)
    
    # Find contours to identify table
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try alternative approach
    if not contours:
        # Try edge detection based approach
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        # If lines detected, create a binary image with the lines
        if lines is not None:
            line_img = np.zeros_like(gray)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
            
            # Find contours again
            contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If still no contours, return entire image as table
    if not contours:
        return original, [(0, img.shape[0])], False
    
    # Find the largest contour (assuming it's the table)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the table
    table_img = original[y:y+h, x:x+w]
    
    # Detect rows in the table
    if len(table_img.shape) == 3:
        table_gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    else:
        table_gray = table_img.copy()
    
    table_thresh = cv2.adaptiveThreshold(table_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (table_img.shape[1]//3, 1))
    horizontal_lines = cv2.morphologyEx(table_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract row positions
    rows = []
    if h_contours:
        # Get y-coordinates of horizontal lines
        h_lines_y = [cv2.boundingRect(c)[1] for c in h_contours]
        h_lines_y.sort()
        
        # Calculate row heights
        for i in range(len(h_lines_y) - 1):
            row_y = h_lines_y[i]
            row_height = h_lines_y[i+1] - h_lines_y[i]
            rows.append((row_y, row_height))
        
        # Add the last row
        if h_lines_y:
            row_y = h_lines_y[-1]
            row_height = table_img.shape[0] - row_y
            rows.append((row_y, row_height))
    
    # If no rows detected, consider the whole table as one row
    if not rows:
        rows = [(0, table_img.shape[0])]
    
    return table_img, rows, True