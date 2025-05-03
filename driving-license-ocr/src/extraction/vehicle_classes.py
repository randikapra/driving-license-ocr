## Improved OCR is dividing two sections. one for date extraction one for vehicle class extraction 
class VehicleClassExtractor:
    """
    Class for handling vehicle class extraction and correction in Sri Lankan driving licenses.
    """
    def __init__(self):
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