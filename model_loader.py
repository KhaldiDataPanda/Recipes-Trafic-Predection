"""
Model Loader Utility
Dynamically loads the best model artifacts from the model/ directory.
Automatically detects and loads model, scaler, and encoder objects.
"""

import os
import glob
import joblib
import warnings
from typing import Dict, Any, Optional, Tuple


class ModelLoader:
    """
    Generic loader for machine learning model artifacts.
    Automatically detects and loads the best model, scaler, and encoder.
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize the model loader.
        
        Args:
            model_dir: Directory containing model artifacts (.pkl files)
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoder = None
        self.metadata = None
        self.category_encoding = None
        
    def load_all(self) -> Tuple[Any, Any, Any, Optional[Dict]]:
        """
        Load all artifacts: model, scaler, encoder, and metadata.
        
        Returns:
            Tuple of (model, scaler, encoder, metadata)
        
        Raises:
            FileNotFoundError: If model directory doesn't exist
            ValueError: If required artifacts are not found
        """
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory '{self.model_dir}' not found")
        
        # Load metadata first if available (contains info about best model)
        self.metadata = self._load_metadata()
        
        # Load model (automatically detect the best one)
        self.model = self._load_model()
        
        # Load scaler
        self.scaler = self._load_scaler()
        
        # Load encoder
        self.encoder = self._load_encoder()
        
        # Generate category encoding mapping from encoder
        self.category_encoding = self._generate_category_encoding()
        
        return self.model, self.scaler, self.encoder, self.metadata
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata if available."""
        metadata_path = os.path.join(self.model_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                print(f"✅ Loaded metadata from {metadata_path}")
                return metadata
            except Exception as e:
                warnings.warn(f"Could not load metadata: {e}")
        return None
    
    def _load_model(self) -> Any:
        """
        Automatically detect and load the best model.
        Looks for model files in the model directory.
        """
        # If metadata exists and has model file info, use it
        if self.metadata and 'best_model_file' in self.metadata:
            model_path = self.metadata['best_model_file']
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    print(f"✅ Loaded model from {model_path}")
                    return model
                except Exception as e:
                    warnings.warn(f"Could not load model from metadata path: {e}")
        
        # Otherwise, search for model files
        model_patterns = [
            "best_model_*.pkl",
            "model_*.pkl",
            "*_model.pkl"
        ]
        
        model_files = []
        for pattern in model_patterns:
            model_files.extend(glob.glob(os.path.join(self.model_dir, pattern)))
        
        if not model_files:
            raise ValueError(f"No model files found in '{self.model_dir}'. Expected files matching: {model_patterns}")
        
        # If multiple models found, try to pick the "best" one
        # Priority: best_model_catboost > best_model_xgboost > best_model_random_forest > others
        priority_keywords = ['catboost', 'xgboost', 'random_forest', 'best']
        
        def get_priority(filepath):
            filename = os.path.basename(filepath).lower()
            for i, keyword in enumerate(priority_keywords):
                if keyword in filename:
                    return i
            return len(priority_keywords)
        
        model_files.sort(key=get_priority)
        model_path = model_files[0]
        
        try:
            model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
            return model
        except Exception as e:
            raise ValueError(f"Error loading model from {model_path}: {e}")
    
    def _load_scaler(self) -> Any:
        """Load the scaler object."""
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from {scaler_path}")
            return scaler
        except Exception as e:
            raise ValueError(f"Error loading scaler from {scaler_path}: {e}")
    
    def _load_encoder(self) -> Any:
        """Load the encoder object."""
        # Try multiple possible encoder filenames
        encoder_filenames = [
            "category_encoder.pkl",
            "target_encoder.pkl",
            "encoder.pkl"
        ]
        
        for filename in encoder_filenames:
            encoder_path = os.path.join(self.model_dir, filename)
            if os.path.exists(encoder_path):
                try:
                    encoder = joblib.load(encoder_path)
                    print(f"✅ Loaded encoder from {encoder_path}")
                    return encoder
                except Exception as e:
                    warnings.warn(f"Could not load encoder from {encoder_path}: {e}")
        
        raise FileNotFoundError(
            f"Encoder not found in '{self.model_dir}'. "
            f"Tried: {encoder_filenames}"
        )
    
    def _generate_category_encoding(self) -> Dict[str, float]:
        """
        Generate category encoding dictionary from the fitted encoder.
        This extracts the actual encoding values learned during training.
        """
        if self.encoder is None:
            raise ValueError("Encoder not loaded. Call load_all() first.")
        
        try:
            # For sklearn TargetEncoder, extract the encodings
            if hasattr(self.encoder, 'encodings_'):
                # encodings_ is a list of arrays (one per feature)
                # Since we only encoded 'category', take the first one
                category_encodings = self.encoder.encodings_[0]
                
                # Get the categories from the encoder
                if hasattr(self.encoder, 'categories_'):
                    categories = self.encoder.categories_[0]
                    
                    # Create mapping
                    encoding_dict = {}
                    for cat, enc in zip(categories, category_encodings):
                        encoding_dict[cat] = float(enc)
                    
                    print(f"✅ Generated category encoding with {len(encoding_dict)} categories")
                    return encoding_dict
            
            # Fallback: if we can't extract encodings, return empty dict
            warnings.warn("Could not extract category encodings from encoder")
            return {}
            
        except Exception as e:
            warnings.warn(f"Error generating category encoding: {e}")
            return {}
    
    def get_category_encoding(self) -> Dict[str, float]:
        """
        Get the category encoding dictionary.
        
        Returns:
            Dictionary mapping category names to encoded values
        """
        if self.category_encoding is None:
            raise ValueError("Category encoding not loaded. Call load_all() first.")
        return self.category_encoding
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'encoder_loaded': self.encoder is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'encoder_type': type(self.encoder).__name__ if self.encoder else None,
        }
        
        if self.metadata:
            info['metadata'] = self.metadata
        
        return info


