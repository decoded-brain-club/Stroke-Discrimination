import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

from feature_functions import extract_all_time_domain_features
from config import ConfigManager # <--- ADDED: Import the actual ConfigManager

# Set MNE logging level to reduce console output
mne.set_log_level('WARNING')

# --- CONFIGURATION (REMOVED SimpleConfigManager) ---

# --- HELPER FUNCTIONS FOR CONFIG-DEPENDENT PATHS AND FILE SCANNING ---

def scan_raw_data_folders(config: ConfigManager) -> Dict[Path, str]:
    """
    Scans the raw data directory specified in config.yaml for EDF files, 
    grouping them by their 'Ischemic' or 'Hemorrhagic' subfolder label.
    """
    file_map = {}
    target_labels = {"Ischemic": "ischemic", "Hemorrhagic": "hemorrhagic"}

    # Construct the full raw data path using the project root and the path from config.yaml
    # config.config['data']['raw_data_path'] is "data/raw_edf"
    raw_data_dir = config.project_root / config.config['data']['raw_data_path']
    
    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory not found at {raw_data_dir}")
        return file_map
        
    for folder_name in target_labels.keys():
        # folder_path would be e.g., '.../data/raw_edf/ischemic' or '.../data/raw_edf/hemorrhagic'
        folder_path = raw_data_dir / folder_name.lower()
        if folder_path.exists() and folder_path.is_dir():
            # Find all .edf files recursively within the subfolder
            for file_path in folder_path.glob("**/*.edf"):
                # Use the full file path as the key, and the class label ("Ischemic" or "Hemorrhagic") as the value
                file_map[file_path] = folder_name
        else:
            print(f"Warning: Data subfolder '{folder_path.name}' not found within {raw_data_dir.name}.")
            
    return file_map

def get_output_filepath(config: ConfigManager) -> Path:
    """
    Calculates the full, versioned path for the output features file.
    It creates the output directory if it doesn't exist.
    """
    # Use the parent directory of one of the processed feature paths from config.yaml
    # e.g., 'data/processed' from "data/processed/train_features.npy"
    processed_dir = config.project_root / Path(config.config['data']['training_features_path']).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Retrieve filtering and epoch parameters from the ConfigManager properties
    low_cut = config.filter_config['low_cutoff']
    high_cut = config.filter_config['high_cutoff']
    t_epoch = int(config.epoch_duration)
    
    # Filename matches the old logic but uses config values
    output_filename = f"features_lgb__{low_cut}-{high_cut}hz_{t_epoch}s_epochs.npz"
    return processed_dir / output_filename


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
        
        # Use the new config property for channels
        raw.pick_channels(config.selected_channels, ordered=False) 
        
        # 3. Set standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # Retrieve filter parameters from the config dictionary
        low_cut = config.filter_config['low_cut_hz']
        high_cut = config.filter_config['high_cut_hz']
        notch_hz = config.filter_config['notch_hz']

        # 4. Filter the data
        raw.filter(low_cut, high_cut, fir_design='firwin')
        raw.notch_filter(notch_hz, fir_design='firwin')

        # 5. Re-reference to common average reference (CAR)
        raw.set_eeg_reference('average', projection=True)
        
        # Use the new config property for epoch duration
        t_epoch = config.epoch_duration 

        # 6. Epoching (creating continuous fixed-length epochs)
        # Define artificial events for continuous epoching
        duration_samples = int(t_epoch * raw.info['sfreq'])
        n_epochs = raw.n_times // duration_samples
        
        events = np.array([[i * duration_samples, 0, 1] for i in range(n_epochs)])
        event_id = 1
        
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id, 
            tmin=0.0, 
            tmax=t_epoch, 
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
    # This function remains unchanged as it uses the epochs object internally
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
    y = np.full(X.shape[0], -1) 
    
    return X, y # y is a placeholder here; the actual label is applied below

def run_feature_extraction(config: ConfigManager):
    """
    Main function to run the full pipeline for all files and save the result.
    """
    all_features = []
    all_labels = []

    # Get the file map using the new helper function
    file_map = scan_raw_data_folders(config)
    output_filepath = get_output_filepath(config)

    print(f"Starting feature extraction. Output will be saved to: {output_filepath}")
    
    # 0 = Ischemic, 1 = Hemorrhagic (Used for LightGBM classification)
    label_map = {"Ischemic": 0, "Hemorrhagic": 1}

    # Iterate over the file map: Path object -> Label string ("Ischemic" or "Hemorrhagic")
    for file_path, label_name in file_map.items():
        print(f"\nProcessing {file_path.name} (Label: {label_name})")
        
        # load_and_preprocess_raw now takes the Path object directly
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
        output_filepath, # Use the calculated output path
        X=X_final, 
        y=y_final,
        # Use the new config property for channels
        feature_names=[
            f"{ch}_{feat}" 
            for ch in config.selected_channels
            for feat in ["ApEn", "FuzzyEn", "Hjorth_Act", "Hjorth_Mob", "Hjorth_Comp", "DFA", "SampEn", "ZeroCross"]
        ]
    )
    print(f"\nSuccessfully saved preprocessed dataset to: {output_filepath.name}")


if __name__ == '__main__':
    # NOTE: You must have your raw EDF files structured in the path specified by config.yaml
    
    print("--- Preprocessing Execution Start ---")
    # --- UPDATED: Instantiate the actual ConfigManager ---
    config = ConfigManager()
    
    # Check the raw data directory
    raw_data_dir = config.project_root / config.config['data']['raw_data_path']
    
    if not raw_data_dir.exists():
        print("\n[IMPORTANT] The raw data folder specified in config.yaml was not found.")
        print(f"Expected path: {raw_data_dir}")
        print("Please ensure your EDF files are placed in this directory structure.")
        
    run_feature_extraction(config)