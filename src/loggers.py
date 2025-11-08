import os
import tempfile
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import mlflow
# We use mlflow.sklearn because LightGBM is typically logged via joblib/pickle 
# or treated as a scikit-learn compatible model for simple logging.
import mlflow.sklearn 
from lightgbm import Booster # Import for type hinting

# Define the target names for the stroke classification model
TARGET_NAMES = ["Ischemic", "Hemorrhagic"]
# If including a control group: TARGET_NAMES = ["Ischemic", "Hemorrhagic", "Control"]

class MLflowTrialLogger:
    """
    MLflow logger specifically designed for logging Optuna trials (child runs)
    and final test results for the ApFu-TPELGBM model.
    """
    
    def __init__(self, config_manager, is_parent_run: bool = False, run_name: str = None):
        """
        Initializes the logger. If is_parent_run is True, it starts the main 
        optimization run. Otherwise, it assumes it will be used for a nested trial.
        """
        self.config = config_manager
        self._is_managing_run = is_parent_run
        
        # Set the tracking URI and experiment name once
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        
        # Start the run context if it's the parent manager
        if self._is_managing_run:
            run = mlflow.start_run(run_name=run_name)
            print(f"MLflow Parent Run Started: {run.info.run_id}")
            self._log_initial_config_params()
        
    def _log_initial_config_params(self):
        """Logs initial configuration parameters and files for the parent run."""
        if not mlflow.active_run():
            return
            
        # Log all configuration parameters from the manager
        mlflow.log_params(self.config.config['data'])
        mlflow.log_params(self.config.config['features'])
        mlflow.log_params(self.config.config['filters'])
        mlflow.log_params(self.config.config['training'])
        
        # Log the config.yaml as an artifact
        try:
            config_path = self.config.project_root / "config.yaml"
            mlflow.log_artifact(str(config_path), "config")
        except Exception as e:
            print(f"Warning: Could not log config.yaml artifact: {e}")

    def log_trial(self, trial_number: int, params: Dict[str, Any], metrics: Dict[str, float]):
        """
        Logs the parameters and metrics for a single Optuna trial as a nested run.
        """
        # This is CRITICAL for TPE optimization: log as a NESTED run
        with mlflow.start_run(run_name=f"Trial-{trial_number}", nested=True) as run:
            
            # Log all optimized hyperparameters for this specific trial
            clean_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(clean_params)
            
            # Log the metrics (e.g., f1_score from cross-validation)
            mlflow.log_metrics(metrics)
            
            # Set a tag to indicate the main metric achieved
            mlflow.set_tag("objective_value", metrics.get(self.config.objective_metric, 'N/A'))
            
            print(f"  > Logged Trial {trial_number} (Run ID: {run.info.run_id}) | {self.config.objective_metric}: {metrics.get(self.config.objective_metric):.4f}")


    def log_final_test_results(self, model: Booster, metrics: Dict[str, float], predictions: List, labels: List):
        """
        Logs final test metrics, model, and detailed artifacts in the active run.
        This should be called inside the FINAL run after TPE is complete.
        """
        if not mlflow.active_run():
             raise RuntimeError("Must call log_final_test_results within an active MLflow run.")
             
        print("\n--- Logging Final Test Results ---")
        
        # Log test metrics
        for name, value in metrics.items():
            mlflow.log_metric(f"final_test_{name}", value)

        # Log Confusion Matrix and Classification Report
        self._log_confusion_matrix(predictions, labels, "final_test_confusion_matrix")
        self._log_classification_report(predictions, labels, "final_test_classification_report")

        # Log the final trained LightGBM model
        try:
            # LightGBM models are easily logged using mlflow.lightgbm or mlflow.sklearn
            # Since LightGBM provides a scikit-learn wrapper, we can use the general sklearn logging
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=self.config.model_architecture,
                registered_model_name=self.config.mlflow_experiment_name
            )
            print(f"Final {self.config.model_architecture} Model Logged.")
        except Exception as e:
            print(f"Error logging LightGBM model: {e}")
            
    def _log_confusion_matrix(self, predictions: List, labels: List, artifact_name: str):
        """Create and log confusion matrix as MLflow artifact, using stroke labels."""
        try:
            cm = confusion_matrix(labels, predictions)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=TARGET_NAMES, 
                        yticklabels=TARGET_NAMES, ax=ax)
            ax.set_xlabel("Predicted Stroke Type")
            ax.set_ylabel("True Stroke Type")
            ax.set_title(f"Confusion Matrix - {artifact_name.replace('_', ' ').title()}")
            
            accuracy = np.trace(cm) / np.sum(cm)
            ax.text(0.5, -0.1, f"Overall Accuracy: {accuracy:.3f}", 
                            transform=ax.transAxes, ha='center', fontsize=12)
            
            plt.tight_layout()
            mlflow.log_figure(fig, f"{artifact_name}.png")
            plt.close(fig)
            
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")

    def _log_classification_report(self, predictions: List, labels: List, artifact_name: str):
        """Log detailed classification report as text artifact, using stroke labels."""
        try:
            report = classification_report(labels, predictions, target_names=TARGET_NAMES, digits=4)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"CLASSIFICATION REPORT FOR {self.config.mlflow_experiment_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, f"{artifact_name}.txt")
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error logging classification report: {e}")

    def close(self):
        """End the MLflow run ONLY if this logger instance started it (i.e., the parent run)."""
        if self._is_managing_run:
            try:
                mlflow.end_run()
                print("MLflow parent run ended successfully.")
            except Exception as e:
                print(f"Error ending MLflow run: {e}")