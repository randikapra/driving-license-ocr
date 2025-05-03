"""
Main implementation of the License OCR pipeline.
"""
## -- Pre-processing Pipeline -- ##
import os
import cv2
import numpy as np
import pytesseract # type: ignore
from pathlib import Path
from .preprocessing.orientation import determine_orientation
from .preprocessing.enhancement import enhance_image_for_ocr
from .preprocessing.table_detection import detect_license_table
def preprocess_license_image(image_path):
    """
    Comprehensive preprocessing pipeline for Sri Lankan driving license images
    that combines table structure detection and OCR confidence for orientation
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Store original for visualization
    original = img.copy()
    
    # Step 1: Determine correct orientation using our combined approach
    oriented_img, angle = determine_orientation(img)
    
    # Step 2: Apply additional image enhancement for better OCR
    enhanced_img = enhance_image_for_ocr(oriented_img)
    
    # Step 3: Detect and extract table region
    table_img, table_roi = detect_license_table(enhanced_img, oriented_img)
    
    return {
        "original": original,
        "oriented": oriented_img,
        "enhanced": enhanced_img,
        "table": table_img,
        "table_roi": table_roi,
        "angle": angle
    }


## --Driving License Data Extraction-- ##

# src/pipeline.py

from src.extraction.full_ocr import ImprovedDrivingLicenseExtractor
from src.extraction.vehicle_classes import VehicleClassExtractor
from src.extraction.dates import DateExtractor
from src.preprocessing.enhancement import enhance_image_for_ocr
from src.preprocessing.table_detection import detect_license_table

def extract_license_data(image_path):
    """Main function to extract data from a driving license image"""
    # Load image
    # Preprocess image
    # Detect table
    # Extract vehicle classes
    # Extract dates
    # Return structured data
    # Initialize the extractor
    extractor = ImprovedDrivingLicenseExtractor()
     
    # Example usage with a single image
    # image_path = "/kaggle/input/test-dataset2/Images/212335.jpg"
    image_path = "../data/raw/298990.jpg"  # Path relative to script location
    image_dir = "../data/raw"              # Path relative to script location
     
    # Process the image
    results = extractor.complete_processing(image_path)
    print("Extracted License Information:")
    print(results)
     
    # Visualize the extraction
    # extractor.visualize_extraction(image_path, save_path="/kaggle/working//extraction_visualization.jpg")
    extractor.visualize_detected_info(image_path)
    
    # Save results to CSV
    results.to_csv("../driving-license-ocr/outputs/tables/license_data_improved.csv", index=False)
     
    # Test batch processing with multiple images
    # Uncomment and modify paths as needed
    '''
    image_paths = [
        "/kaggle/input/test-dataset2/Images/230559.jpg",
        "/kaggle/input/test-dataset2/Images/other_image.jpg"
    ]
    batch_results = extractor.process_batch(
        image_paths, 
        output_dir="/kaggle/working/processed_images"
    )
    print("Batch Processing Results:")
    print(batch_results)
    '''








# """
# Main implementation of the License OCR pipeline.
# """
# import logging
# from pathlib import Path
# from typing import Dict, List, Optional, Union, Any, Tuple

# import cv2 # type: ignore
# import numpy as np # type: ignore
# import pandas as pd # type: ignore

# from src.utils.config import get_config # type: ignore
# from src.preprocessing.alignment import align_image # type: ignore
# from src.preprocessing.enhancement import enhance_image # type: ignore
# from src.preprocessing.segmentation import extract_region_of_interest # type: ignore
# from src.ocr.engine import perform_ocr # type: ignore
# from src.ocr.postprocessing import clean_text # type: ignore
# from src.extraction.vehicle_class import extract_vehicle_classes # type: ignore
# from src.extraction.date_parser import extract_dates, associate_dates_with_classes # type: ignore
# from src.utils.image_utils import load_image, save_image # type: ignore
# from src.utils.visualization import visualize_results # type: ignore


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# class LicenseOCRResult:
#     """Container for the results of the License OCR pipeline."""
    
#     def __init__(
#         self,
#         vehicle_classes: List[str],
#         start_dates: List[str],
#         expiry_dates: List[str],
#         confidence_scores: Optional[List[float]] = None
#     ):
#         """
#         Initialize the result object.
        
#         Args:
#             vehicle_classes: List of extracted vehicle classes
#             start_dates: List of start dates (same length as vehicle_classes)
#             expiry_dates: List of expiry dates (same length as vehicle_classes)
#             confidence_scores: Optional list of confidence scores
#         """
#         if len(start_dates) != len(vehicle_classes) or len(expiry_dates) != len(vehicle_classes):
#             raise ValueError("All result lists must have the same length")
            
#         self.vehicle_classes = vehicle_classes
#         self.start_dates = start_dates
#         self.expiry_dates = expiry_dates
#         self.confidence_scores = confidence_scores or [1.0] * len(vehicle_classes)
        
#     def to_dict(self) -> Dict[str, List[Any]]:
#         """Convert result to dictionary."""
#         return {
#             "vehicle_class": self.vehicle_classes,
#             "start_date": self.start_dates,
#             "expiry_date": self.expiry_dates,
#             "confidence": self.confidence_scores
#         }
        
#     def to_dataframe(self) -> pd.DataFrame:
#         """Convert result to pandas DataFrame."""
#         return pd.DataFrame(self.to_dict())
    
#     def to_csv(self, output_path: str) -> None:
#         """Save result to CSV file."""
#         self.to_dataframe().to_csv(output_path, index=False)
        
#     def __str__(self) -> str:
#         """String representation of the result."""
#         result_str = "Extracted License Information:\n"
#         result_str += "-" * 50 + "\n"
#         result_str += "| {:^10} | {:^12} | {:^12} |\n".format(
#             "Class", "Start Date", "Expiry Date"
#         )
#         result_str += "-" * 50 + "\n"
        
#         for cls, start, expiry in zip(self.vehicle_classes, self.start_dates, self.expiry_dates):
#             result_str += "| {:^10} | {:^12} | {:^12} |\n".format(cls, start, expiry)
            
#         result_str += "-" * 50 + "\n"
#         return result_str
        

# class LicenseOCRPipeline:
#     """
#     Main pipeline for extracting information from Sri Lankan driving licenses.
#     """
    
#     def __init__(self, config_path: Optional[str] = None):
#         """
#         Initialize the pipeline.
        
#         Args:
#             config_path: Optional path to configuration file
#         """
#         self.config = get_config(config_path)
#         logger.info("License OCR pipeline initialized")
        
#     def process(
#         self, 
#         image_path: Union[str, Path], 
#         save_debug: bool = False,
#         debug_dir: Optional[Union[str, Path]] = None
#     ) -> LicenseOCRResult:
#         """
#         Process a single license image.
        
#         Args:
#             image_path: Path to the license image
#             save_debug: Whether to save intermediate results for debugging
#             debug_dir: Directory to save debug images (if save_debug is True)
            
#         Returns:
#             LicenseOCRResult containing extracted information
#         """
#         image_path = Path(image_path)
#         logger.info(f"Processing image: {image_path}")
        
#         # Prepare debug directory if needed
#         if save_debug:
#             debug_dir = Path(debug_dir or "debug")
#             debug_dir.mkdir(exist_ok=True)
#             image_debug_dir = debug_dir / image_path.stem
#             image_debug_dir.mkdir(exist_ok=True)
#         else:
#             image_debug_dir = None
            
#         # Step 1: Load and preprocess image
#         image = load_image(image_path)
#         if image is None:
#             raise ValueError(f"Failed to load image: {image_path}")
        
#         # Step 2: Preprocess the image
#         preprocessed = self._preprocess_image(image, image_debug_dir)
        
#         # Step 3: Perform OCR
#         ocr_text = self._perform_ocr(preprocessed, image_debug_dir)
        
#         # Step 4: Extract vehicle classes and dates
#         result = self._extract_information(ocr_text)
        
#         # Visualize results if debug is enabled
#         if save_debug and image_debug_dir:
#             vis_image = visualize_results(image, result)
#             save_image(vis_image, image_debug_dir / "final_result.jpg")
            
#         logger.info(f"Completed processing: {image_path}")
#         return result
        
#     def _preprocess_image(
#         self, 
#         image: np.ndarray,
#         debug_dir: Optional[Path] = None
#     ) -> np.ndarray:
#         """
#         Preprocess the image for better OCR results.
        
#         Args:
#             image: Input image
#             debug_dir: Directory to save debug images
            
#         Returns:
#             Preprocessed image
#         """
#         logger.debug("Preprocessing image")
        
#         # Resize image
#         resize_width = self.config["preprocessing"]["resize_width"]
#         scale = resize_width / image.shape[1]
#         resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
#         if debug_dir:
#             save_image(resized, debug_dir / "01_resized.jpg")
        
#         # Align image
#         if self.config["preprocessing"]["deskew"]:
#             aligned = align_image(resized)
#             if debug_dir:
#                 save_image(aligned, debug_dir / "02_aligned.jpg")
#         else:
#             aligned = resized
            
#         # Extract region of interest
#         roi = extract_region_of_interest(aligned)
#         if debug_dir:
#             save_image(roi, debug_dir / "03_roi.jpg")
            
#         # Enhance image for better OCR
#         if self.config["preprocessing"]["contrast_enhancement"]:
#             enhanced = enhance_image(
#                 roi, 
#                 denoise=self.config["preprocessing"]["denoise"],
#                 method=self.config["preprocessing"]["threshold_method"]
#             )
#             if debug_dir:
#                 save_image(enhanced, debug_dir / "04_enhanced.jpg")
#         else:
#             enhanced = roi
            
#         return enhanced
        
#     def _perform_ocr(
#         self, 
#         image: np.ndarray,
#         debug_dir: Optional[Path] = None
#     ) -> str:
#         """
#         Perform OCR on the preprocessed image.
        
#         Args:
#             image: Preprocessed image
#             debug_dir: Directory to save debug info
            
#         Returns:
#             Extracted text
#         """
#         logger.debug("Performing OCR")
        
#         ocr_config = self.config["ocr"]
#         raw_text = perform_ocr(
#             image,
#             lang=ocr_config["lang"],
#             config=ocr_config["config"]
#         )
        
#         # Clean up text
#         cleaned_text = clean_text(raw_text)
        
#         if debug_dir:
#             with open(debug_dir / "ocr_raw.txt", "w") as f:
#                 f.write(raw_text)
#             with open(debug_dir / "ocr_cleaned.txt", "w") as f:
#                 f.write(cleaned_text)
                
#         return cleaned_text
        
#     def _extract_information(self, text: str) -> LicenseOCRResult:
#         """
#         Extract vehicle classes and validity dates from OCR text.
        
#         Args:
#             text: OCR text
            
#         Returns:
#             LicenseOCRResult containing extracted information
#         """
#         logger.debug("Extracting information")
        
#         # Extract vehicle classes
#         vehicle_classes = extract_vehicle_classes(
#             text, 
#             known_classes=self.config["extraction"]["vehicle_classes"]
#         )
        
#         # Extract dates
#         all_dates = extract_dates(
#             text,
#             date_formats=self.config["extraction"]["date_formats"],
#             locale=self.config["extraction"]["default_locale"]
#         )
        
#         # Associate dates with vehicle classes
#         class_date_pairs = associate_dates_with_classes(text, vehicle_classes, all_dates)
        
#         # Prepare result lists
#         result_classes = []
#         result_start_dates = []
#         result_expiry_dates = []
#         result_confidences = []
        
#         for cls, (start_date, expiry_date, confidence) in class_date_pairs.items():
#             result_classes.append(cls)
#             result_start_dates.append(start_date)
#             result_expiry_dates.append(expiry_date)
#             result_confidences.append(confidence)
            
#         return LicenseOCRResult(
#             vehicle_classes=result_classes,
#             start_dates=result_start_dates,
#             expiry_dates=result_expiry_dates,
#             confidence_scores=result_confidences
#         )
        
#     def process_batch(
#         self, 
#         image_paths: List[Union[str, Path]],
#         output_dir: Optional[Union[str, Path]] = None,
#         save_debug: bool = False
#     ) -> Dict[str, LicenseOCRResult]:
#         """
#         Process a batch of license images.
        
#         Args:
#             image_paths: List of paths to license images
#             output_dir: Directory to save results
#             save_debug: Whether to save intermediate results for debugging
            
#         Returns:
#             Dictionary mapping image paths to LicenseOCRResult objects
#         """
#         results = {}
#         output_dir = Path(output_dir) if output_dir else None
        
#         if output_dir:
#             output_dir.mkdir(exist_ok=True)
            
#         for img_path in image_paths:
#             img_path = Path(img_path)
#             try:
#                 result = self.process(
#                     img_path,
#                     save_debug=save_debug,
#                     debug_dir=output_dir / "debug" if output_dir else None
#                 )
                
#                 results[str(img_path)] = result
                
#                 if output_dir:
#                     result.to_csv(output_dir / f"{img_path.stem}_result.csv")
                    
#             except Exception as e:
#                 logger.error(f"Error processing {img_path}: {str(e)}")
#                 results[str(img_path)] = None
                
#         # Combine all results into a single CSV if output_dir is provided
#         if output_dir:
#             all_data = []
#             for img_path, result in results.items():
#                 if result:
#                     df = result.to_dataframe()
#                     df["source_image"] = img_path
#                     all_data.append(df)
                    
#             if all_data:
#                 combined_df = pd.concat(all_data, ignore_index=True)
#                 combined_df.to_csv(output_dir / "all_results.csv", index=False)
                
#         return results