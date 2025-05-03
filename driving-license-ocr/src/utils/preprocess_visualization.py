import matplotlib.pyplot as plt # type: ignore
import cv2

def visualize_preprocessing(results, title=None):
    """
    Visualize the results of preprocessing steps
    """
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(2, 2, 1)
    if len(results["original"].shape) > 2:
        plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(results["original"], cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Oriented image
    plt.subplot(2, 2, 2)
    if len(results["oriented"].shape) > 2:
        plt.imshow(cv2.cvtColor(results["oriented"], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(results["oriented"], cmap='gray')
    plt.title(f"Oriented Image (Angle: {results['angle']:.1f}°)")
    plt.axis('off')
    
    # Enhanced image
    plt.subplot(2, 2, 3)
    plt.imshow(results["enhanced"], cmap='gray')
    plt.title("Enhanced Image for OCR")
    plt.axis('off')
    
    # Table detection
    plt.subplot(2, 2, 4)
    if results["table_roi"] is not None:
        plt.imshow(results["table_roi"], cmap='gray')
        plt.title("Extracted Table Region")
    else:
        plt.imshow(results["table"], cmap='gray')
        plt.title("Table Region Detection")
    plt.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return plt


import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_preprocessing(original_img, enhanced_img, table_img=None, rows=None):
    """
    Visualize the preprocessing steps with the original and enhanced images.
    
    Args:
        original_img: Original input image
        enhanced_img: Enhanced image after preprocessing
        table_img: Detected table image (optional)
        rows: List of (y, height) for each detected row (optional)
        
    Returns:
        None (displays visualization)
    """
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Show enhanced image
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')
    
    # Show detected table if available
    if table_img is not None:
        plt.subplot(2, 2, 3)
        table_vis = table_img.copy()
        
        # Draw detected rows if available
        if rows:
            for y, h in rows:
                cv2.line(table_vis, (0, y), (table_vis.shape[1], y), (0, 255, 0), 2)
                cv2.line(table_vis, (0, y+h), (table_vis.shape[1], y+h), (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(table_vis, cv2.COLOR_BGR2RGB))
        plt.title('Detected Table with Rows')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_extraction_results(table_img, extraction_results):
    """
    Visualize the extraction results on the table image.
    
    Args:
        table_img: Detected table image
        extraction_results: Dictionary of extracted data
        
    Returns:
        None (displays visualization)
    """
    plt.figure(figsize=(15, 5))
    
    # Show table image with results
    result_img = table_img.copy()
    
    # Annotate with extracted information
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    
    # Add extraction results as text overlay
    for i, (cls, dates) in enumerate(extraction_results.items()):
        start_date, end_date = dates
        text = f"Class: {cls}, Valid: {start_date} to {end_date}"
        cv2.putText(result_img, text, (10, y_offset), font, font_scale, (0, 255, 0), 2)
        y_offset += 30
    
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Extraction Results')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()