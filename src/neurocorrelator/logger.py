import mlflow
from omegaconf import DictConfig, OmegaConf


class MLflowLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.logging.mlflow.project)

    def log_params(self):
        # Логируем всю конфигурацию
        mlflow.log_params(OmegaConf.to_container(self.cfg, resolve=True))

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def __enter__(self):
        self.run = mlflow.start_run()
        self.log_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
