import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt # type: ignore
from pipeline import preprocess_license_image
from preprocess_visualization import visualize_preprocessing

def process_single_image(image_path):
    """Process a single image and show results"""
    print(f"Processing {image_path}...")
    
    # Preprocess image
    results = preprocess_license_image(image_path)
    
    if results is None:
        print(f"Error processing image.")
        return None
    
    # Visualize results
    plt_obj = visualize_preprocessing(results, f"Preprocessing: {Path(image_path).name}")
    plt_obj.show()
    
    print(f"Completed processing.")
    return results

def process_directory(image_dir, output_dir=None, max_images=5):
    """Process all images in directory and save/show results"""
    # Get all image paths
    image_paths = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.png')) + \
                  list(Path(image_dir).glob('*.jpeg'))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process limited number of images
    for i, image_path in enumerate(image_paths[:max_images]):
        image_name = image_path.name
        print(f"Processing {image_name}...")
        
        try:
            # Preprocess image
            results = preprocess_license_image(image_path)
            
            if results is None:
                print(f"Error processing {image_name}, skipping.")
                continue
                
            # Visualize results
            plt_obj = visualize_preprocessing(results, f"Preprocessing: {image_name}")
            
            # Display
            plt_obj.show()
            
            # Save if output directory specified
            if output_dir:
                output_path = os.path.join(output_dir, f"processed_{image_name}.png")
                plt_obj.savefig(output_path)
                print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    print("Preprocessing complete!")


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




# import os
# import cv2
# import json
# import numpy as np
# from datetime import datetime

# def load_image(image_path):
#     """
#     Load and validate an image from a given path.
    
#     Args:
#         image_path: Path to the input image
        
#     Returns:
#         Loaded image as numpy array, or None if loading fails
#     """
#     if not os.path.exists(image_path):
#         print(f"Error: Image not found at {image_path}")
#         return None
        
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Error: Could not read image at {image_path}")
#             return None
#         return img
#     except Exception as e:
#         print(f"Error loading image: {str(e)}")
#         return None

# def save_results(output_path, extraction_results, metadata=None):
#     """
#     Save extraction results to a JSON file.
    
#     Args:
#         output_path: Path to save the results JSON
#         extraction_results: Dictionary of extraction results
#         metadata: Optional metadata to include in the results
        
#     Returns:
#         True if successful, False otherwise
#     """
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
