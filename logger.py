# -*- coding: utf-8 -*-
import os
from datetime import datetime
import uuid
import torch
import json
import yaml
import numbers
from loguru import logger
import mlflow  
import re
# from collections import defaultdict

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

def average_scalar_fields_only(entries):
    if not entries:
        return {}
    
    accumulators = {}
    
    for entry in entries:
        data = entry["data"]
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if key not in accumulators:
                    accumulators[key] = []
                accumulators[key].append(float(value))
            # Skip lists, dicts, etc.
    
    # Compute averages
    averages = {}
    for key, values in accumulators.items():
        averages[key] = sum(values) / len(values)
    
    return averages


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
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M_')+str(uuid.uuid4()) if not self.use_ml_flow else self.mlflow_run.info.run_id
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
        if self.use_ml_flow:
            mlflow.log_artifact(config_path)
        self.logger.info(f"Config saved to {config_path}")

        config_path_yaml = os.path.join(self.logs_dir, "config.yaml")
        with open(config_path_yaml, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        if self.use_ml_flow:
            mlflow.log_artifact(config_path_yaml)
        self.logger.info(f"Config saved to {config_path_yaml}")

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

    def _sanitize_for_metric(self, s: str) -> str:
        s = str(s).strip().replace(" ", "_")
        s = re.sub(r"[^a-zA-Z0-9._\-]+", "_", s)
        return s


    def _normalize_value_for_json(self, v):
        """
        Convert values to JSON-friendly types, preserving structure (lists/dicts).
        numbers -> float; tensors -> float (scalar) or list; lists/tuples/dicts -> recurse.
        Strings оставляем как есть (парсинг делаем в numeric-extractor).
        """
        try:
            if isinstance(v, (str, bool)) or v is None:
                return v
            if isinstance(v, numbers.Number):
                return float(v)
            if torch.is_tensor(v):
                if v.numel() == 1:
                    return float(v.item())
                return v.detach().cpu().tolist()
            if isinstance(v, (list, tuple)):
                return [self._normalize_value_for_json(x) for x in v]
            if isinstance(v, dict):
                return {str(k): self._normalize_value_for_json(val) for k, val in v.items()}
            s = str(v)
            return s if len(s) <= 512 else s[:509] + "..."
        except Exception:
            return None

    def _iter_numeric_leaf_values(self, v):
        """Yield floats from scalars, tensors, lists/tuples, dicts; also parse JSON-looking strings."""
        try:
            if isinstance(v, numbers.Number):
                yield float(v); return
            if torch.is_tensor(v):
                if v.numel() == 1:
                    yield float(v.item()); return
                for x in v.detach().cpu().flatten().tolist():
                    if isinstance(x, numbers.Number):
                        yield float(x)
                return
            if isinstance(v, (list, tuple)):
                for x in v:
                    yield from self._iter_numeric_leaf_values(x)
                return
            if isinstance(v, dict):
                for x in v.values():
                    yield from self._iter_numeric_leaf_values(x)
                return
            if isinstance(v, str):
                s = v.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                    import json as _json
                    try:
                        parsed = _json.loads(s)
                        yield from self._iter_numeric_leaf_values(parsed)
                    except Exception:
                        return
                return
        except Exception:
            return

    def _collect_metadata(self, model):
        """
        Собираем *.metadata/_meta/meta у модулей и параметров.
        Возвращаем:
          - entries: список словарей вида
              { "parent_path": str, "source": "module.meta|param.meta|...", "data": {...} }
          - aggregates: { parent_path: { meta_key: {count, mean, min, max} } }
        """
        entries: list[dict] = []
        numeric_buckets: dict[tuple[str, str], list[float]] = {}

        def push_entry(parent_path: str, source: str, meta_dict: dict):
            if not isinstance(meta_dict, dict):
                return
            json_entry = {
                "parent_path": parent_path,
                "source": source,
                "data": {str(k): self._normalize_value_for_json(v) for k, v in meta_dict.items()},
            }
            entries.append(json_entry)
            for k, v in meta_dict.items():
                key = (parent_path, str(k))
                for num in self._iter_numeric_leaf_values(v):
                    numeric_buckets.setdefault(key, []).append(num)

        for mod_name, mod in model.named_modules():
            parent_path = mod_name if mod_name else model.__class__.__name__
            for attr in ("metadata", "_meta", "meta"):
                if hasattr(mod, attr):
                    push_entry(parent_path, source=f"module.{attr}", meta_dict=getattr(mod, attr))

            named_params = getattr(mod, 'named_parameters', None)
            if callable(named_params):
                for p_name, p in named_params(recurse=False):
                    p_path = f"{parent_path}.{p_name}"
                    for attr in ("metadata", "_meta", "meta"):
                        if hasattr(p, attr):
                            push_entry(p_path, source=f"param.{attr}", meta_dict=getattr(p, attr))

        # aggregates: dict[str, dict[str, list]] = {}
        # for (parent_path, meta_key), vals in numeric_buckets.items():
        #     if not vals:
        #         continue
        #     parent_dict = aggregates.setdefault(parent_path, {})
        #     parent_dict[meta_key] = vals
        #     # parent_dict[meta_key] =  {
        #     #     "mean": float(sum(vals) / len(vals)),
        #     #     "min": float(min(vals)),
        #     #     "max": float(max(vals)),
        #     # }
        return entries

    def _save_metadata_artifact(self, payload: dict, step: int):
        """Save raw metadata + aggregates to artifact (local + MLflow)."""
        os.makedirs(self.logs_dir, exist_ok=True)
        metadata_dir = os.path.join(self.logs_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        sample_path = os.path.join(metadata_dir, f"metadata_step{step:06d}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.info(f"Saved metadata to {sample_path}")
        if self.use_ml_flow and mlflow is not None:
            try:
                mlflow.log_artifact(sample_path, artifact_path="metadata")
                aggr = payload.get("aggregates", {})  
                metrics = {}
                for parent_path, meta_dict in aggr.items():
                    parent_s = self._sanitize_for_metric(parent_path)
                    for meta_key, stat in meta_dict.items():
                        key_s = self._sanitize_for_metric(meta_key)
                        for stat_name in ("mean", "min", "max", "count"):
                            if stat_name in stat:
                                metrics[f"meta_agg/{parent_s}/{key_s}_{stat_name}"] = float(stat[stat_name])
                if metrics:
                    mlflow.log_metrics(metrics)
                self.logger.info("Metadata artifact + aggregate metrics logged to MLflow.")
            except Exception as e:
                self.logger.info(f"Failed to log metadata to MLflow: {e}")

    def collect_and_log_metadata(self, model, step, tokens_per_iter):
        
        entries = self._collect_metadata(model)
        payload = {
            "step": int(step),
            "tokens_per_iter": int(tokens_per_iter),
            "entries": entries,
        }
        self._save_metadata_artifact(payload, step=step)

        average = average_scalar_fields_only(entries)
        mlflow.log_metrics(average)

    def save_sample_artifact(self, sample_dict: dict, step: int, multi: bool = False):
        """Save JSON with sample(s) locally and (if enabled) to MLflow under artifact_path='samples'."""
        os.makedirs(self.logs_dir, exist_ok=True)
        fname = (f"samples_step{step:06d}.json" if multi else f"sample_step{step:06d}.json")
        samples_dir = os.path.join(self.logs_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        sample_path = os.path.join(samples_dir, fname)
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_dict, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved sample(s) to {sample_path}")

        if self.use_ml_flow and mlflow is not None:
            try:
                mlflow.log_artifact(sample_path, artifact_path="samples")
                self.logger.info("Sample artifact logged to MLflow under 'samples/'.")
            except Exception as e:
                self.logger.info(f"Failed to log sample to MLflow: {e}")
    
    def save_model_architecture(self, model):
        """Save/emit print(model) either locally (model.txt) or to MLflow if enabled."""
        text = ""
        try:
            text = str(model)  
        except Exception as e:
            text = f"<failed to stringify model: {e}>"

        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            path = os.path.join(self.logs_dir, "model.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            self.logger.info(f"Model architecture saved to {path}")
        except Exception as e:
            self.logger.info(f"Failed to save model architecture locally: {e}")
            path = None

        if self.use_ml_flow and mlflow is not None:
            try:
                if hasattr(mlflow, "log_text"):
                    mlflow.log_text(text, artifact_file="model.txt")
                    self.logger.info("Model architecture logged to MLflow (log_text).")
                elif path is not None:
                    mlflow.log_artifact(path, artifact_path="model")
                    self.logger.info("Model architecture logged to MLflow (artifact).")
            except Exception as e:
                self.logger.info(f"Failed to log model architecture to MLflow: {e}")
