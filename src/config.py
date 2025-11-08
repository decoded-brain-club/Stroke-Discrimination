import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the configuration manager.
        """
        # 1. Determine the path to config.yaml
        if config_path is None:
            # Assumes this script is in a subdirectory (e.g., 'utils')
            self.project_root = Path(__file__).resolve().parent.parent 
            config_path = self.project_root / "config.yaml"
        else:
            self.project_root = Path(config_path).parent.parent # Simplified assumption

        if not os.path.exists(config_path):
            # Log the error before raising
            logging.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # 2. Load the YAML content
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 3. Setup basic logging
        self._setup_logging()
        logging.info("Configuration loaded successfully.")

    def _setup_logging(self):
        """Setup basic logging configuration based on config.yaml."""
        # Use log level from config if available, otherwise INFO
        log_level = self.config['logging'].get('log_level', 'INFO').upper() 
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths (from config) to absolute paths based on project root."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str(self.project_root / path_obj)
    
    # --- DATA PATHS (Adapted for ApFu Features) ---
    @property
    def raw_data_path(self) -> str:
        return self._resolve_path(self.config['data']['raw_data_path'])
    
    @property
    def training_features_path(self) -> str:
        # Renamed to feature paths to reflect ApFu feature matrices
        return self._resolve_path(self.config['data']['training_features_path'])
    
    @property
    def validation_features_path(self) -> str:
        return self._resolve_path(self.config['data']['validation_features_path'])
    
    @property
    def testing_features_path(self) -> str:
        return self._resolve_path(self.config['data']['testing_features_path'])
    
    # --- MODEL & SAVE PATHS ---
    @property
    def model_save_path(self) -> str:
        return self._resolve_path(self.config['model']['save_path'])
    
    @property
    def final_model_filename(self) -> str:
        return self.config['model']['model_filename']
    
    @property
    def best_trial_model_filename(self) -> str:
        # Use best_model_filename from the YAML for clarity
        return self.config['model']['best_model_filename']
    
    @property
    def model_architecture(self) -> str:
        return self.config['model']['architecture']
    
    # --- LOGGING PROPERTIES ---
    @property
    def log_path(self) -> str:
        return self._resolve_path(self.config['logging']['log_path'])
    
    @property
    def mlflow_enabled(self) -> bool:
        return self.config['logging']['mlflow_enabled']
    
    @property
    def mlflow_experiment_name(self) -> str:
        return self.config['logging']['mlflow_experiment_name']

    @property
    def mlflow_tracking_uri(self) -> str:
        return self.config['logging']['mlflow_tracking_uri']

    # --- TPE/TRAINING PROPERTIES (Adapted for Optuna) ---
    @property
    def tpe_n_trials(self) -> int:
        return self.config['training']['tpe_n_trials']
    
    @property
    def n_splits_cv(self) -> int:
        return self.config['training']['n_splits_cv']

    @property
    def seed(self) -> int:
        return self.config['training']['seed']
    
    @property
    def objective_metric(self) -> str:
        return self.config['training']['objective_metric']
    
    @property
    def tpe_search_bounds(self) -> Dict[str, Union[int, float]]:
        return self.config['training']['search_bounds']

    # --- FEATURE PROPERTIES ---
    @property
    def feature_type(self) -> str:
        return self.config['features']['type']

    @property
    def m_embed_dim(self) -> int:
        return self.config['features']['m_embed_dim']

    @property
    def r_tolerance_factor(self) -> float:
        return self.config['features']['r_tolerance_factor']

    # --- PROCESSING PROPERTIES ---
    @property
    def epoch_duration(self) -> float:
        return self.config['processing']['epoch_duration']
    
    @property
    def target_sfreq(self) -> int:
        return self.config['processing']['target_sfreq']
    
    @property
    def selected_channels(self) -> List[str]:
        return self.config['processing']['selected_channels']
    
    @property
    def filter_config(self) -> Dict[str, float]:
        return self.config['filters']
    
    @property
    def data_split_config(self) -> Dict[str, float]:
        return self.config['data_split']

    # --- UTILITY PATH METHODS ---
    def get_full_model_path(self) -> str:
        """Get the full path to save the final model"""
        return os.path.join(self.model_save_path, self.final_model_filename)
    
    def get_full_best_trial_model_path(self) -> str:
        """Get the full path to save the best trial model found by TPE"""
        return os.path.join(self.model_save_path, self.best_trial_model_filename)

# --- Singleton Implementation ---
_config_instance: Optional[ConfigManager] = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """Get global config instance (Singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance