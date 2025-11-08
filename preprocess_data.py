import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

from feature_functions import extract_all_time_domain_features

# Set MNE logging level to reduce console output
mne.set_log_level('WARNING')

# --- CONFIGURATION (Simplified placeholder to match the structure) ---
# NOTE: In a real system, this would load the full config.yaml
class SimpleConfigManager:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = "Z:\Data\TUH"
        self.processed_dir = "Z:\Data\TUH\processed_features_discrim"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Preprocessing parameters (from your assumed config)
        self.sfreq = 256.0 # Expected sampling frequency
        self.low_cut_hz = 0.5
        self.high_cut_hz = 45.0
        self.notch_hz = [50.0] # Assuming European power line frequency
        self.t_epoch = 4.0 # Epoch length in seconds (e.g., 4s segments)
        
        # EEG Channel Information (adjust to your specific electrode set)
        # Standard 10-20 or 10-10 systems are common
        self.eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

    def _scan_data_folders(self) -> Dict[Path, str]:
        """
        Scans 'ischemic' and 'hemorrhagic' subfolders for EDF files.
        Returns a dictionary mapping absolute file paths to their class label.
        """
        file_map = {}
        target_labels = {"Ischemic": "ischemic", "Hemorrhagic": "hemorrhagic"}
        
        for folder_name, label_name in target_labels.items():
            folder_path = self.data_dir / folder_name.lower()
            if folder_path.exists() and folder_path.is_dir():
                # Find all .edf files recursively within the subfolder
                for file_path in folder_path.glob("*.edf"):
                    file_map[file_path] = label_name
            else:
                print(f"Warning: Data subfolder '{folder_path.name}' not found.")
                
        return file_map

    def get_raw_filepath(self, filename: str) -> Path:
        return self.data_dir / filename

    @property
    def output_filepath(self) -> Path:
        # Save name based on filtering parameters
        return self.processed_dir / f"features_lgb__{self.low_cut_hz}-{self.high_cut_hz}hz_{int(self.t_epoch)}s_epochs.npz"

# --- MAIN PREPROCESSING FUNCTIONALITY ---

def load_and_preprocess_raw(file_path: Path, config: ConfigManager) -> Optional[mne.Epochs]:
    """
    Loads one EDF file, applies filtering, re-referencing, and epoching.
    """
    try:
        # 1. Load data from EDF
        raw = mne.io.read_raw_edf(str(file_path), preload=True)
        
        # 2. Rename channels and select relevant EEG channels
        raw.rename_channels(lambda x: x.replace(' ', '').upper()) # Clean channel names
        
        # Drop non-EEG channels if they exist, and select only the standard set
        raw.pick_channels(config.eeg_channels, ordered=False)
        
        # 3. Set standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # 4. Filter the data
        raw.filter(config.low_cut_hz, config.high_cut_hz, fir_design='firwin')
        raw.notch_filter(config.notch_hz, fir_design='firwin')

        # 5. Re-reference to common average reference (CAR)
        raw.set_eeg_reference('average', projection=True)
        
        # 6. Epoching (creating continuous fixed-length epochs)
        # Define artificial events for continuous epoching
        duration_samples = int(config.t_epoch * raw.info['sfreq'])
        n_epochs = raw.n_times // duration_samples
        
        events = np.array([[i * duration_samples, 0, 1] for i in range(n_epochs)])
        event_id = 1
        
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id, 
            tmin=0.0, 
            tmax=config.t_epoch, 
            baseline=None, 
            preload=True, 
            # Reject based on peak-to-peak amplitude (150 uV is common for adult EEG)
            reject=dict(eeg=150e-6) 
        )
        
        print(f"File: {file_path.name} | Original epochs: {n_epochs} | Rejected epochs: {n_epochs - len(epochs)}")
        return epochs

    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return None

def extract_features_from_epochs(epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterates through epochs, extracts features, and creates the feature matrix X and label vector y.
    """
    feature_list = []
    
    # Epochs are stored as a 3D array: (n_epochs, n_channels, n_samples)
    data = epochs.get_data(units='uV') # Get data in microvolts (uV)
    
    sfreq = epochs.info['sfreq']

    for i in range(data.shape[0]):
        # data[i] is (n_channels, n_samples)
        epoch_features = extract_all_time_domain_features(data[i], sfreq)
        feature_list.append(epoch_features)

    # Convert to NumPy arrays
    X = np.stack(feature_list)
    
    # Create the label vector (all epochs from this file have the same label)
    # The true label (Ischemic/Hemorrhagic) will be set in the main script.
    y = np.full(X.shape[0], -1) 
    
    return X, y # y is a placeholder here; the actual label is applied below

def run_feature_extraction(config: SimpleConfigManager):
    """
    Main function to run the full pipeline for all files and save the result.
    """
    all_features = []
    all_labels = []

    print(f"Starting feature extraction. Output will be saved to: {config.output_filepath}")
    
    # 0 = Ischemic, 1 = Hemorrhagic (Used for LightGBM classification)
    label_map = {"Ischemic": 0, "Hemorrhagic": 1}

    for filename, label_name in config.data_files.items():
        file_path = config.get_raw_filepath(filename)

        if not file_path.exists():
            print(f"Skipping {filename}: File not found at {file_path}")
            continue

        print(f"\nProcessing {filename} (Label: {label_name})")
        epochs = load_and_preprocess_raw(file_path, config)

        if epochs:
            X_file, _ = extract_features_from_epochs(epochs)
            y_file = np.full(X_file.shape[0], label_map[label_name])
            
            all_features.append(X_file)
            all_labels.append(y_file)
            
            print(f"Extracted {X_file.shape[0]} epochs, {X_file.shape[1]} features.")

    if not all_features:
        print("No features were extracted. Check data file paths and file formats.")
        return

    # Concatenate all subjects' features and labels
    X_final = np.concatenate(all_features, axis=0)
    y_final = np.concatenate(all_labels, axis=0)

    print("\n--- Summary ---")
    print(f"Total Epochs Processed (X): {X_final.shape}")
    print(f"Total Labels (y): {y_final.shape}")
    print(f"Class counts (0=Ischemic, 1=Hemorrhagic): {np.unique(y_final, return_counts=True)}")

    # Save the final dataset
    np.savez_compressed(
        config.output_filepath, 
        X=X_final, 
        y=y_final,
        # Store metadata about the features for later inspection
        feature_names=[
            f"{ch}_{feat}" 
            for ch in config.eeg_channels
            for feat in ["ApEn", "FuzzyEn", "Hjorth_Act", "Hjorth_Mob", "Hjorth_Comp", "DFA", "SampEn", "ZeroCross"]
        ]
    )
    print(f"\nSuccessfully saved preprocessed dataset to: {config.output_filepath.name}")


if __name__ == '__main__':
    # NOTE: You must create a 'data' folder in the same directory as this script 
    # and place your EDF files inside it for this script to run successfully.
    
    # --- Example Usage (REQUIRES YOUR EDF FILES) ---
    print("--- Preprocessing Execution Start ---")
    config = SimpleConfigManager()
    
    # Check if the data folder exists and warn the user
    if not config.data_dir.exists():
        print("\n[IMPORTANT] The 'data' folder was not found.")
        print("Please create a folder named 'data' next to this script and place your EDF files inside.")
        print("Using placeholder files: ", list(config.data_files.keys()))
        # Create dummy files for a dry run (will still fail to load, but shows the path check)
        config.data_dir.mkdir(exist_ok=True)
        for fname in config.data_files.keys():
            (config.data_dir / fname).touch()
            
    run_feature_extraction(config)