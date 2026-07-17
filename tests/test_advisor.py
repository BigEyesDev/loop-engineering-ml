"""
Tests for claude_advisor.py

Fast tests (no API):
  python tests/test_advisor.py

Include live API call:
  python tests/test_advisor.py --live
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from claude_advisor import extract_yaml, request_config_update, ADJUSTABLE_FIELDS, FIELD_TYPES


# ---------------------------------------------------------------------------
# Unit tests — no API call, instant
# ---------------------------------------------------------------------------

class TestExtractYaml(unittest.TestCase):

    def test_clean_yaml(self):
        text = "learning_rate: 0.001\nbatch_size: 32\nfreeze_backbone: false\n"
        result = extract_yaml(text)
        self.assertEqual(result["learning_rate"], 0.001)
        self.assertEqual(result["batch_size"], 32)
        self.assertFalse(result["freeze_backbone"])

    def test_fenced_yaml_block(self):
        text = (
            "Sure, here is the config:\n"
            "```yaml\n"
            "learning_rate: 5.0e-05\n"
            "batch_size: 64\n"
            "freeze_backbone: false\n"
            "```\n"
            "This should help break the plateau."
        )
        result = extract_yaml(text)
        self.assertAlmostEqual(result["learning_rate"], 5e-5)
        self.assertEqual(result["batch_size"], 64)

    def test_fenced_no_language_tag(self):
        text = "Here:\n```\nweight_decay: 0.005\nbatch_size: 32\n```"
        result = extract_yaml(text)
        self.assertEqual(result["weight_decay"], 0.005)

    def test_mixed_text_then_yaml(self):
        text = (
            "I recommend unfreezing the backbone.\n\n"
            "learning_rate: 1.0e-05\n"
            "batch_size: 16\n"
            "freeze_backbone: false\n"
            "warmup_steps: 200\n"
        )
        result = extract_yaml(text)
        self.assertAlmostEqual(result["learning_rate"], 1e-5)
        self.assertFalse(result["freeze_backbone"])

    def test_raises_on_garbage(self):
        with self.assertRaises(ValueError):
            extract_yaml("This is just a sentence with no YAML at all.")

    def test_raises_on_empty(self):
        with self.assertRaises((ValueError, Exception)):
            extract_yaml("")


class TestAdjustableFieldsFilter(unittest.TestCase):
    """Verify that only ADJUSTABLE_FIELDS survive the whitelist, not hallucinated keys."""

    def _apply_filter(self, advisor_response: dict, base_config: dict) -> dict:
        final_config = dict(base_config)
        for field in ADJUSTABLE_FIELDS:
            if field in advisor_response:
                value = advisor_response[field]
                if field in FIELD_TYPES:
                    value = FIELD_TYPES[field](value)
                final_config[field] = value
        return final_config

    def test_only_adjustable_fields_written(self):
        base = {
            "learning_rate": 2e-5, "batch_size": 16, "freeze_backbone": True,
            "warmup_steps": 100, "weight_decay": 0.01,
            "model": "google/vit-base-patch16-224", "num_epochs": 15, "target_f1": 0.95,
        }
        advisor_response = {
            "learning_rate": 1e-5,
            "batch_size": 32,
            "num_epochs": 999,    # hallucinated — should be blocked
            "target_f1": 0.5,     # hallucinated — should be blocked
            "new_field": "oops",  # completely invented
        }
        result = self._apply_filter(advisor_response, base)
        self.assertEqual(result["learning_rate"], 1e-5)
        self.assertEqual(result["batch_size"], 32)
        self.assertEqual(result["num_epochs"], 15)    # unchanged
        self.assertEqual(result["target_f1"], 0.95)  # unchanged
        self.assertNotIn("new_field", result)

    def test_string_numeric_fields_are_coerced(self):
        """GPT-5 returned learning_rate as a quoted string — must become float."""
        base = {"learning_rate": 2e-5, "batch_size": 16, "freeze_backbone": True,
                "warmup_steps": 100, "weight_decay": 0.01}
        advisor_response = {
            "learning_rate": "2e-05",   # string — what GPT-5 actually returned
            "batch_size": "32",         # string int
            "warmup_steps": "0",        # string zero
        }
        result = self._apply_filter(advisor_response, base)
        self.assertIsInstance(result["learning_rate"], float)
        self.assertAlmostEqual(result["learning_rate"], 2e-5)
        self.assertIsInstance(result["batch_size"], int)
        self.assertEqual(result["batch_size"], 32)
        self.assertIsInstance(result["warmup_steps"], int)
        self.assertEqual(result["warmup_steps"], 0)

    def test_freeze_backbone_bool(self):
        base = {"freeze_backbone": True, "learning_rate": 2e-5, "batch_size": 16,
                "warmup_steps": 100, "weight_decay": 0.01}
        advisor_response = {"freeze_backbone": False, "learning_rate": 5e-6}
        result = self._apply_filter(advisor_response, base)
        self.assertFalse(result["freeze_backbone"])
        self.assertAlmostEqual(result["learning_rate"], 5e-6)


# ---------------------------------------------------------------------------
# Live integration test — real API call, opt-in only
# ---------------------------------------------------------------------------

class TestLiveAdvisorCall(unittest.TestCase):

    MINIMAL_CONFIG = {
        "advisor_model": "openai/gpt-5",
        "model": "google/vit-base-patch16-224",
        "dataset": "blanchon/EuroSAT_RGB",
        "num_epochs": 15,
        "target_f1": 0.95,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "freeze_backbone": True,
        "max_claude_calls": 3,
        "min_delta": 0.005,
        "plateau_window": 2,
        "num_classes": 10,
    }

    MINIMAL_HISTORY = [
        {"epoch": 1, "f1_macro": 0.73, "train_loss": 1.67},
        {"epoch": 2, "f1_macro": 0.82, "train_loss": 0.95},
        {"epoch": 3, "f1_macro": 0.91, "train_loss": 0.31},
        {"epoch": 4, "f1_macro": 0.91, "train_loss": 0.29},  # plateau
    ]

    def _config_path(self):
        import tempfile, yaml
        # Mirror real layout: <root>/config/image_eurosat.yaml → <root>/logs/advisor_calls/
        # A flat /tmp file would resolve logs to /logs (permission denied).
        tmpdir = tempfile.mkdtemp()
        config_dir = os.path.join(tmpdir, "config")
        os.makedirs(config_dir)
        config_path = os.path.join(config_dir, "image_eurosat.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.MINIMAL_CONFIG, f)
        return config_path

    def test_live_call_returns_valid_yaml_and_adjustable_fields(self):
        if "OPENAI_API_KEY" not in os.environ:
            self.skipTest("OPENAI_API_KEY not set — skipping live test")

        import tempfile
        config_path = self._config_path()

        tokens = request_config_update(
            config=self.MINIMAL_CONFIG,
            epoch_history=self.MINIMAL_HISTORY,
            current_epoch=4,
            config_path=config_path,
        )

        # Tokens used > 0 means the call succeeded and content was returned
        self.assertGreater(tokens, 0, "API returned 0 tokens — call likely failed")

        # Config file was rewritten with valid YAML
        import yaml
        with open(config_path) as f:
            written = yaml.safe_load(f)

        self.assertIsInstance(written, dict, "Written config is not a dict")

        # Protected fields must survive unchanged
        self.assertEqual(written["model"], self.MINIMAL_CONFIG["model"])
        self.assertEqual(written["num_epochs"], self.MINIMAL_CONFIG["num_epochs"])
        self.assertEqual(written["target_f1"], self.MINIMAL_CONFIG["target_f1"])

        # At least one adjustable field must be present (advisor wrote something)
        has_adjustable = any(f in written for f in ADJUSTABLE_FIELDS)
        self.assertTrue(has_adjustable, f"No adjustable fields found in written config: {written}")

        print(f"\n  Tokens used: {tokens}")
        print(f"  Written config: {written}")


if __name__ == "__main__":
    live = "--live" in sys.argv
    if "--live" in sys.argv:
        sys.argv.remove("--live")

    if not live:
        # Skip the live test class entirely when not requested
        suite = unittest.TestLoader().loadTestsFromTestCase(TestExtractYaml)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdjustableFieldsFilter))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    else:
        # Load .env so OPENAI_API_KEY is available
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(unittest.TestLoader().loadTestsFromModule(
            sys.modules[__name__]
        ))

    sys.exit(0 if result.wasSuccessful() else 1)
