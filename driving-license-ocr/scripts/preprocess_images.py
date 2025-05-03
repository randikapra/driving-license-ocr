import os
import sys
# Add the parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import process_single_image, process_directory

if __name__ == "__main__":
    # Path to a single image or directory of images
    image_path = "../data/raw/298990.jpg"  # Path relative to script location
    image_dir = "../data/raw"              # Path relative to script location
    
    # Process single image
    results = process_single_image(image_path)
    
    # Or process directory
    process_directory(image_dir, output_dir="../data/processed", max_images=5)