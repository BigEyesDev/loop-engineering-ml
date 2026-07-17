import yaml


_CONFIG_TYPES: dict = {
    "learning_rate": float,
    "batch_size": int,
    "warmup_steps": int,
    "weight_decay": float,
    "num_epochs": int,
    "num_classes": int,
    "max_claude_calls": int,
    "plateau_window": int,
}


def load_config(config_path: str) -> dict:
    """Load and return a YAML config file as a dict from the given path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for key, cast in _CONFIG_TYPES.items():
        if key in config:
            config[key] = cast(config[key])
    return config


def is_plateau(f1_history: list, min_delta: float, window: int = 2) -> bool:
    """Return True if F1 hasn't improved by more than min_delta over the last `window` epochs."""
    if len(f1_history) < window:
        return False
    recent = f1_history[-window:]
    return (recent[-1] - recent[0]) <= min_delta


def setup_mlflow_experiment(experiment_name: str) -> None:
    """Create or activate the named MLflow experiment for the current session."""
    import mlflow
    mlflow.set_experiment(experiment_name)
