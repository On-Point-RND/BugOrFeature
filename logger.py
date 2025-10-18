import os
import uuid
import torch
import json
from loguru import logger
import mlflow  

def _format_value(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    elif torch.is_tensor(v) and v.numel() == 1:
        return f"{v.item():.4f}"
    elif hasattr(v, 'item'):  # e.g., NumPy scalars
        try:
            return f"{v.item():.4f}"  # fixed typo: was "4sf"
        except (AttributeError, TypeError):
            pass
    return str(v)


class Logger:
    def __init__(self, config, use_ml_flow=False):
        self.config = config
        self.use_ml_flow = use_ml_flow

        # --- MLflow setup ---
        if self.use_ml_flow:

            mlflow.set_tracking_uri(config['logging'].get('mlflow_tracking_uri', 'mlruns'))  # optional URI
            mlflow.set_experiment(config['logging'].get('experiment_name', 'default'))
            self.mlflow_run = mlflow.start_run(run_name=config.get('run_name', None))
            self._log_config_to_mlflow()
        else:
            self.mlflow_run = None

        # --- Local logging setup ---
        self.run_id = str(uuid.uuid4()) if not self.use_ml_flow else self.mlflow_run.info.run_id
        self.logs_dir = os.path.join(config['logging']['output_dir'], self.run_id)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.log_file = os.path.join(self.logs_dir, "log_file.log")
        self.logger = logger
        self.logger.add(self.log_file)

        self.logger.info(f"Log dir created at {self.logs_dir}")
        if self.use_ml_flow:
            self.logger.info(f"MLflow run ID: {self.mlflow_run.info.run_id}")
            self.logger.info(f"MLflow run URL: {mlflow.get_artifact_uri()}")  # or construct UI URL if known

        self.save_every = config['logging']['save_every']
        self._save_config()

    def _log_config_to_mlflow(self):
        """Flatten config and log as MLflow parameters (max 100 params per call)."""
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            return dict(items)

        flat_config = flatten_dict(self.config)
        # MLflow allows up to 100 params per log_params call
        items = list(flat_config.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i:i+100]))

    def _save_config(self):
        config_path = os.path.join(self.logs_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Config saved to {config_path}")
        if self.use_ml_flow:
            mlflow.log_artifact(config_path)

    def log(self, **kwargs):
        # Log to console/file
        msg = ", ".join(f"{k}={_format_value(v)}" for k, v in kwargs.items())
        self.logger.info(msg)

        # Log to MLflow
        if self.use_ml_flow:
            # MLflow expects numeric values; filter out non-numeric if needed
            mlflow_metrics = {}
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    mlflow_metrics[k] = v
                elif torch.is_tensor(v) and v.numel() == 1:
                    mlflow_metrics[k] = v.item()
                elif hasattr(v, 'item'):
                    try:
                        mlflow_metrics[k] = v.item()
                    except (AttributeError, TypeError):
                        continue
            if mlflow_metrics:
                mlflow.log_metrics(mlflow_metrics)

    def info(self, info):
        self.logger.info(info)

    def save_model(self, model, step, tag=None):
        filename = f"model_step{step:06d}.pt" if tag is None else f"model_{tag}.pt"
        model_path = os.path.join(self.logs_dir, filename)
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved at step {step} to {model_path}")
        if self.use_ml_flow:
            mlflow.log_artifact(model_path)

    def check_save_step(self, step):
        return (step + 1) % self.save_every == 0

    def __del__(self):
     if self.use_ml_flow and self.mlflow_run:
        # Safely end MLflow run only if mlflow is still available
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except (AttributeError, ImportError, TypeError):
            # mlflow may be unloaded during shutdown — ignore
            pass