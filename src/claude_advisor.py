import json
import os
from pathlib import Path
import yaml
from openai import OpenAI


PROTECTED_FIELDS = {"model", "dataset", "num_epochs", "target_f1"}
ADJUSTABLE_FIELDS = {"learning_rate", "batch_size", "warmup_steps", "weight_decay", "freeze_backbone"}

# Models sometimes return numeric values as quoted strings — coerce to correct types.
FIELD_TYPES: dict = {
    "learning_rate": float,
    "batch_size": int,
    "warmup_steps": int,
    "weight_decay": float,
}

ADVISOR_SYSTEM_PROMPT = (
    "You are an ML training agent with write access to the training config file. "
    "Make one targeted adjustment per call. "
    "Adjustable fields: learning_rate, batch_size, warmup_steps, weight_decay, "
    "freeze or unfreeze layers and backbone. "
    "Respond ONLY with the full updated config as valid YAML. No explanation. No preamble. No markdown fences."
)

ADVISOR_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_training_payload(config: dict, epoch_history: list, current_epoch: int) -> dict:
    """Build the minimal JSON payload for the advisor; returns dict with history and context."""
    return {
        "task": "image_classification",
        "model": config["model"],
        "dataset": config["dataset"],
        "current_epoch": current_epoch,
        "history": epoch_history,
        "freeze_backbone": config.get("freeze_backbone", False),
        "target_f1": config["target_f1"],
        "remaining_epochs": config["num_epochs"] - current_epoch,
    }


def extract_yaml(text: str) -> dict:
    """Parse YAML from model response — handles raw YAML, fenced blocks, and mixed text+YAML."""
    # Try raw parse first (clean response)
    try:
        result = yaml.safe_load(text)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass

    # Try extracting from ```yaml ... ``` or ``` ... ``` fences
    import re
    fenced = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            result = yaml.safe_load(fenced.group(1))
            if isinstance(result, dict):
                return result
        except yaml.YAMLError:
            pass

    # Last resort: find first line that looks like a YAML key and parse from there
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*\w[\w_]*\s*:", line):
            try:
                result = yaml.safe_load("\n".join(lines[i:]))
                if isinstance(result, dict):
                    return result
            except yaml.YAMLError:
                pass
            break

    raise ValueError(f"Could not extract valid YAML dict from response:\n{text[:300]}")


def load_advisor_principles(config_path: str) -> str:
    """Load advisor_principles.md from the same directory as config_path; returns content or empty string."""
    principles_path = Path(config_path).parent / "advisor_principles.md"
    if principles_path.exists():
        return principles_path.read_text()
    return ""


def request_config_update(config: dict, epoch_history: list, current_epoch: int, config_path: str) -> int:
    """Call the advisor via OpenRouter, parse and validate YAML response, write to config_path; returns tokens used."""
    payload = build_training_payload(config, epoch_history, current_epoch)
    principles = load_advisor_principles(config_path)
    advisor_model = config.get("advisor_model", ADVISOR_MODEL)

    user_content = json.dumps(payload)
    if principles:
        user_content = f"Principles to follow:\n{principles}\n\nTraining context:\n{user_content}"

    try:
        if advisor_model.startswith("openai/") and "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
            base_url = "https://api.openai.com/v1"
            model_id = advisor_model.split("/", 1)[1]  # strip "openai/" prefix
            print(f"  Using advisor model: {model_id} (direct OpenAI API)")
        else:
            api_key = os.environ["OPENROUTER_API_KEY"]
            base_url = OPENROUTER_BASE_URL
            model_id = advisor_model
            print(f"  Using advisor model: {model_id} (OpenRouter)")

        client = OpenAI(api_key=api_key, base_url=base_url)
        # GPT-5 and o-series models use max_completion_tokens; all others use max_tokens
        token_kwarg = (
            {"max_completion_tokens": 4000}
            if advisor_model.startswith("openai/") and "OPENAI_API_KEY" in os.environ
            else {"max_tokens": 4000}
        )
        response = client.chat.completions.create(
            model=model_id,
            **token_kwarg,
            messages=[
                {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        tokens_used = response.usage.total_tokens
        raw_text = response.choices[0].message.content
        advisor_response = extract_yaml(raw_text)

        final_config = dict(config)
        for field in ADJUSTABLE_FIELDS:
            if field in advisor_response:
                value = advisor_response[field]
                if field in FIELD_TYPES:
                    value = FIELD_TYPES[field](value)
                final_config[field] = value

        with open(config_path, "w") as f:
            yaml.dump(final_config, f, default_flow_style=False)

        # Save Gemini's output for inspection
        log_dir = Path(config_path).parent.parent / "logs" / "advisor_calls"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"epoch_{current_epoch:02d}.yaml"
        with open(log_path, "w") as f:
            yaml.dump({
                "epoch": current_epoch,
                "tokens_used": tokens_used,
                "raw_gemini_response": advisor_response,
                "final_config_written": final_config,
                "fields_changed": {
                    k: {"before": config.get(k), "after": final_config.get(k)}
                    for k in ADJUSTABLE_FIELDS
                    if config.get(k) != final_config.get(k)
                },
            }, f, default_flow_style=False)

        print(f"  Gemini wrote → {log_path}")
        print(f"  Fields changed: { {k: v['after'] for k, v in yaml.safe_load(log_path.read_text())['fields_changed'].items()} }")

        return tokens_used

    except Exception as exc:
        print(f"Advisor call failed: {exc}")
        return 0
