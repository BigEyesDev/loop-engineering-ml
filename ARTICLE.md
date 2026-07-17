# Loop Engineering: Teaching an AI to Fine-Tune Itself

**A technical walkthrough of building an autonomous ML training loop where an LLM advisor rewrites hyperparameters in real time — with actual results from a ViT trained on EuroSAT satellite imagery.**

---

## TL;DR

We built a ViT fine-tuning loop that calls Gemini autonomously when training plateaus. Gemini writes hyperparameter changes directly to a YAML config file. The loop reads the file on the next epoch. No human intervention between start and finish.

We ran five experiments across four models (Gemini 2.5 Flash, Gemini 3.1 Pro, GPT-5) and three prompt strategies to answer two questions most loop engineering articles skip: *how much does the prompt shape the advisor's reasoning?* And *does the model matter?*

The answers surprised us. The vague prompt outperformed both the prescriptive recipe and the principled context document — because the LLM already knew what to do. Helping it too much constrained it. Warning it too much paralysed it. And the bigger model (Gemini 3.1 Pro) found a creative but wrong strategy, while two smaller models with the same vague prompt immediately found the right one.

**Best result:** F1=0.9791 in 10 epochs, 1 call, 596 tokens — with the simplest prompt and the baseline model.
**Most efficient path to target:** GPT-5, 1 call, early stop at F1=0.9757, 10 epochs.

---

## The Problem With Traditional Fine-Tuning

When you fine-tune a model, you sit in the middle of a decision cycle:

```
Train → Evaluate → Notice plateau → Decide what to change → Adjust → Repeat
```

That "notice" and "decide" step is you. Every time. You are the loop. You watch TensorBoard, you lower the learning rate, you unfreeze layers, you rebuild the optimizer. Your expertise is the bottleneck.

Loop engineering asks: what if the loop ran that decision cycle itself?

> *"I don't prompt Claude anymore. I have loops running that prompt Claude and figure out what to do. My job is to write loops."*
> — Boris Cherny, Head of Claude Code at Anthropic

---

## What We Built

**Stack:**
- Model: `google/vit-base-patch16-224` (86M parameters)
- Dataset: EuroSAT RGB — 10-class satellite land cover classification
- Advisor: Gemini 2.5 Flash via OpenRouter
- Tracking: MLflow
- Config: YAML file on disk

**The core loop:**

```
Epoch N starts
  → Reload config from disk
  → Train one epoch
  → Evaluate F1 on validation set
  → If F1 plateau detected → call Gemini
    → Gemini writes new YAML to disk
  → Log everything to MLflow
Epoch N+1 starts (with Gemini's changes already live)
```

The human runs `python3 src/train.py`. The human reads the MLflow results. Everything in between is autonomous.

---

## The Architecture: Five Files

```
loop/
├── config/
│   └── image_eurosat.yaml      ← Gemini writes here
├── src/
│   ├── utils.py                ← config loading, plateau detection, MLflow setup
│   ├── data_loader.py          ← HuggingFace → PyTorch Dataset bridge
│   ├── evaluate.py             ← model.eval() + metric computation
│   ├── claude_advisor.py       ← the Gemini call + config write
│   └── train.py                ← the main orchestrator loop
```

Each file has one job. No file knows more than it needs to.

---

## Design Pattern 1: Config-as-Interface

The central architectural decision: **Gemini does not return a suggestion. Gemini writes the config.**

The naive version:
```python
# Gemini says: "lower learning rate to 5e-6"
# Your code parses that text and applies it
suggestion = call_gemini(context)
apply_suggestion(suggestion, optimizer)   # fragile — what if format varies?
```

Our version:
```python
# Gemini receives the current config
# Gemini returns the full updated config as YAML
# We write it to disk
# The next epoch reads it — same as always
with open(config_path, "w") as f:
    yaml.dump(updated_config, f)
```

The training loop doesn't know Gemini was called. It just reads the config file at the start of every epoch, same as it always does. Gemini's agency is expressed entirely through that file.

---

## Design Pattern 2: The Prompt — and Why It Matters More Than You Think

We ran three experiments with three different prompts. The prompt turned out to be the most
consequential variable in the system — more than the model, more than the plateau threshold.

### The Prompt Spectrum

```
Prescriptive ←————————————————————————→ Principled
  Give the recipe          Describe consequences, not rules
```

**Experiment 1 — Prescriptive (gave the recipe):**
```
Adjustable fields: learning_rate, batch_size, warmup_steps, weight_decay,
freeze_backbone (bool — set to false to unfreeze the full ViT backbone for fine-tuning;
pair this with a lower learning_rate such as 5e-6).
```

**Experiment 2 — Vague (named the levers, no guidance):**
```
Adjustable fields: learning_rate, batch_size, warmup_steps, weight_decay,
freeze or unfreeze layers and backbone.
```

**Experiment 3 — Principled (context document with consequences):**
```
Adjustable fields: learning_rate, batch_size, warmup_steps, weight_decay,
freeze or unfreeze layers and backbone.
+ advisor_principles.md loaded as context
```

The principles document describes *consequences*, not instructions:
> *"Unfreezing adds ~85M parameters to training overnight. A learning rate appropriate
> for 7K parameters can be catastrophically large for 85M — it will overwrite pretrained
> ImageNet representations before the model has a chance to adapt them."*

Gemini draws its own conclusion. You just made sure it had the relevant physics.

### System Prompt (all three runs)

```
You are an ML training agent with write access to the training config file.
Make one targeted adjustment per call.
Adjustable fields: learning_rate, batch_size, warmup_steps, weight_decay,
freeze or unfreeze layers and backbone.
Respond ONLY with the full updated config as valid YAML. No explanation. No preamble. No markdown fences.
```

**Why this prompt works:**
- One adjustment per call — preserves the causal relationship between change and effect
- Output format constraint — "ONLY valid YAML, no markdown fences" — forces machine-parseable output

### User Payload (what we send each call)

```json
{
  "task": "image_classification",
  "model": "google/vit-base-patch16-224",
  "dataset": "blanchon/EuroSAT_RGB",
  "current_epoch": 10,
  "history": [
    {"epoch": 1,  "f1": 0.7243, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 2,  "f1": 0.7891, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 3,  "f1": 0.8312, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 4,  "f1": 0.8601, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 5,  "f1": 0.8745, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 6,  "f1": 0.8834, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 7,  "f1": 0.8921, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 8,  "f1": 0.8991, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 9,  "f1": 0.9028, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true},
    {"epoch": 10, "f1": 0.9046, "lr": 2e-05, "batch_size": 16, "freeze_backbone": true}
  ],
  "freeze_backbone": true,
  "target_f1": 0.95,
  "remaining_epochs": 5
}
```

**~300 tokens total.** No code. No logs. No architecture description. Nothing beyond what Gemini needs to make one good decision.

### Gemini's Response (raw output we parse)

```yaml
batch_size: 32
dataset: blanchon/EuroSAT_RGB
freeze_backbone: true
learning_rate: 2.0e-05
max_claude_calls: 3
min_delta: 0.005
model: google/vit-base-patch16-224
num_classes: 10
num_epochs: 15
plateau_window: 2
target_f1: 0.95
warmup_steps: 100
weight_decay: 0.01
```

Plain YAML. We parse it, extract only the adjustable fields, write the full config to disk. Done.

**How a string becomes a dict — two things happening here:**

First, the system prompt ends with `"Respond ONLY with the full updated config as valid YAML. No explanation. No preamble. No markdown fences."` — that's what forces raw YAML out instead of a conversational response.

Second, `yaml.safe_load(response.choices[0].message.content)` converts the YAML string into a Python dict. Gemini returns text. `yaml.safe_load` is the bridge.

**The hallucination problem — and why it doesn't matter:**

We never send Gemini the full config. The payload only contains `model`, `dataset`, `history`, `target_f1`, `remaining_epochs`. Yet Gemini returns a full config including fields like `num_classes`, `max_claude_calls`, `min_delta` — values it was never shown. It infers or guesses them.

This is why the `ADJUSTABLE_FIELDS` filter is not optional:

```python
final_config = dict(config)        # starts from the REAL current config
for field in ADJUSTABLE_FIELDS:    # only copies the 5 allowed fields
    if field in advisor_response:
        final_config[field] = advisor_response[field]
```

If Gemini hallucinated `num_classes: 8` instead of 10, it doesn't matter — `num_classes` is never copied. The real value survives untouched. The filter is what makes Gemini's partial hallucinations harmless.

---

## Design Pattern 3: Protected Fields

Gemini has write access to config — but not to everything.

```python
PROTECTED_FIELDS  = {"model", "dataset", "num_epochs", "target_f1"}
ADJUSTABLE_FIELDS = {"learning_rate", "batch_size", "warmup_steps", "weight_decay", "freeze_backbone"}

# After Gemini writes its response:
final_config = dict(current_config)           # start from current
for field in ADJUSTABLE_FIELDS:               # only copy adjustable fields
    if field in gemini_response:
        final_config[field] = gemini_response[field]
# Protected fields are never overwritten
```

Gemini cannot change the model, the dataset, or the target. It can only turn the knobs you explicitly exposed. This is the constraint that makes autonomous operation safe — you define the surface area of Gemini's agency.

---

## Design Pattern 4: Plateau Detection

```python
def is_plateau(f1_history: list, min_delta: float = 0.005, window: int = 2) -> bool:
    if len(f1_history) < window:
        return False
    recent = f1_history[-window:]
    return (recent[-1] - recent[0]) <= min_delta
```

**Why `window=2`?** One flat epoch is noise. Two consecutive flat epochs is a structural signal — the optimizer has found a local ceiling at the current hyperparameters. That is when an external advisor's perspective has value.

**Why `min_delta=0.005`?** F1 fluctuates by this amount naturally due to minibatch stochasticity. Triggering below this threshold produces false positives — calling Gemini when training is actually progressing fine. 0.005 is the noise floor.

---

## Design Pattern 5: Backbone Freezing as a Loop Engineering Trick

Without freezing, ViT achieves F1=0.98 on EuroSAT in epoch 1. The pretrained backbone is so powerful, and EuroSAT is simple enough, that the model converges trivially. Gemini is never called. The loop never exercises its core behavior.

With `freeze_backbone: true` at the start, only the 7,690-parameter classifier head trains. The 86M backbone parameters are locked. The classifier learns on top of general ImageNet features — good, but not optimised for EuroSAT. This causes a natural plateau around F1=0.90, triggering the advisor calls.

The freezing strategy also mirrors real production behavior: freeze → train head → plateau → advisor suggests unfreezing → full fine-tune. This is the standard progressive unfreezing recipe, here executed autonomously.

```python
def apply_backbone_freeze(model, freeze: bool) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze

# When freeze_backbone changes between epochs:
if new_freeze != active_freeze:
    active_freeze = new_freeze
    apply_backbone_freeze(model, active_freeze)
    optimizer = build_optimizer(model, lr, weight_decay)  # rebuild — new param set
    print(f"Backbone {'frozen' if active_freeze else 'unfrozen'} — optimizer rebuilt")
```

---

## The Three Experiments — Side by Side

All three runs started from identical config: `freeze_backbone: true`, `LR: 2e-5`, `batch: 16`.

### Run 1 — Prescriptive Prompt

| Epoch | F1 | Event |
|-------|----|-------|
| 1–9 | 0.72 → 0.90 | Frozen backbone, classifier head training |
| 10 | 0.9046 | **Gemini call 1 (674 tokens) → batch_size: 32** |
| 11 | 0.9061 | **Gemini call 2 (1,188 tokens) → warmup_steps: 500** |
| 12 | 0.9079 | **Gemini call 3 (768 tokens) → freeze_backbone: false + LR: 5e-6** |
| 13 | **0.9688** | Backbone unfrozen — full ViT fine-tunes |

**Best F1: 0.9688 | Calls: 3 | Tokens: ~2,630**

Gemini followed the checklist it was given: tried conservative levers first (batch size, warmup), then finally unfroze. The recipe in the prompt shaped its strategy.

---

### Run 2 — Vague Prompt (no recipe)

| Epoch | F1 | Event |
|-------|----|-------|
| 1–8 | 0.70 → 0.90 | Frozen backbone |
| 9 | 0.9033 | **Gemini call 1 (596 tokens) → freeze_backbone: false** |
| 10 | **0.9791** | Backbone unfrozen — full ViT fine-tunes |

**Best F1: 0.9791 | Calls: 1 | Tokens: 596**

Without the recipe, Gemini immediately identified the frozen backbone as the bottleneck. It went straight to the architectural lever — no warmup changes, no batch tweaks. Better result, fewer calls, fewer tokens.

**The catch:** Gemini kept LR at `2e-5` when unfreezing and also zeroed `weight_decay` and `warmup_steps`. It got away with it on EuroSAT — a clean, balanced dataset. On a noisier dataset, keeping LR at `2e-5` while unfreezing 85M parameters could have caused catastrophic forgetting: the pretrained ImageNet representations overwritten before the model adapts them to the new task.

Why does high LR destroy pretrained weights? The update rule is:
```
new_weight = old_weight - (learning_rate × gradient)
```
With a large LR and large gradient, each step moves weights significantly. Across 85M parameters per batch, the pretrained configuration gets overwritten fast — before the model has learned what's useful for EuroSAT. A lower LR gives the optimizer time to layer new task features on top of existing representations rather than replacing them.

---

### Run 3 — Principled Prompt (context document)

Gemini receives the vague prompt plus `advisor_principles.md` prepended to the user message. The document describes consequences of each decision — emphasising the risks of unfreezing at high LR and catastrophic forgetting.

| Epoch | F1 | Event |
|-------|----|-------|
| 1–8 | 0.71 → 0.90 | Frozen backbone |
| 9 | 0.9041 | **Gemini call 1 (1,113 tokens) → warmup_steps: 0** |
| 10 | 0.9061 | **Gemini call 2 (1,167 tokens) → batch_size: 32** |
| 11 | 0.9096 | **Gemini call 3 (1,205 tokens) → batch_size: 64** |
| 12–15 | 0.9099 → 0.9128 | Budget exhausted. Backbone still frozen. |

**Best F1: 0.9128 | Calls: 3 | Tokens: ~3,485**

Final config written by Gemini:
```yaml
freeze_backbone: true    # never touched
batch_size: 64           # 16 → 32 → 64, one call each
warmup_steps: 0          # zeroed on call 1
learning_rate: 2.0e-05   # unchanged throughout
```

**Gemini never unfroze the backbone.** It used all 3 calls scaling batch_size and ran out of budget. The principles document described catastrophic forgetting so vividly that Gemini avoided the one lever that would have broken the plateau. It played it safe — and safe was wrong.

---

### Run 4 — Gemini 3.1 Pro (same vague prompt)

Same prompt as Run 2 — but a different model. Gemini 3.1 Pro is larger, slower, and more deliberate than Flash.

| Epoch | F1 | Event |
|-------|----|-------|
| 1–8 | 0.71 → 0.90 | Frozen backbone |
| 9 | 0.9033 | **Call 1 → batch_size: 1** (SGD-like gradient noise) |
| 11 | 0.9055 | **Call 2 → warmup_steps: 0, weight_decay: 0** |
| 12 | 0.9082 | **Call 3 → warmup_steps: 0, weight_decay: 0** (repeat) |
| 15 | 0.9360 | Run ends. Backbone never unfrozen. |

**Best F1: 0.9360 | Calls: 3 | Tokens: ~5,100**

Pro chose `batch_size: 1` — an unconventional move that forces the model to update on every single sample, introducing gradient noise that can escape shallow plateaus. It's a valid strategy. It just wasn't the right strategy here: the bottleneck was the frozen backbone, not noisy gradients. Pro found a creative lever; it found the wrong one.

---

### Run 5 — GPT-5 (same vague prompt, direct OpenAI API)

GPT-5 required two infrastructure fixes before it would run: `max_completion_tokens` instead of `max_tokens` (GPT-5 uses a different parameter name), and type coercion at config load time (GPT-5 occasionally returns numeric values as quoted strings, which PyTorch's AdamW rejects). Both bugs were caught by the test suite before any training time was lost.

| Epoch | F1 | Event |
|-------|----|-------|
| 1–8 | 0.68 → 0.90 | Frozen backbone |
| 9 | 0.9006 | **Call 1 (2,863 tokens) → freeze_backbone: false** |
| 10 | **0.9757** | Backbone unfrozen — full ViT fine-tunes |
| — | — | **Target F1 0.95 reached. Early stop.** |

**Best F1: 0.9757 | Calls: 1 | Tokens: 2,863**

GPT-5's reasoning: saw the frozen backbone plateau, unfroze immediately, same as Gemini 2.5 Flash in Run 2. One call, one correct decision, done.

**The high-LR-when-unfreezing concern, revisited:** GPT-5 also kept LR at `2e-5` when unfreezing — the same choice Run 2 made. We had expected catastrophic forgetting. It didn't happen. EuroSAT is clean and balanced enough that `2e-5` on 85M params is survivable. The principles document's emphasis on this risk was correct in theory, but overcautious in practice for this specific dataset.

**The one-change-at-a-time tension:** The "one targeted adjustment per call" rule is correct for independent parameters. But unfreeze + LR reduction are coupled: unfreezing is only fully safe when LR is simultaneously reduced. The one-change rule made it structurally impossible for the advisor to do both at once. The loop's recovery mechanism is the answer — if LR had caused instability, the next plateau call would have fixed it.

---

## What the Advisors Decided

### Run 1 — Call 3 (prescribed recipe)
**Decision:** `freeze_backbone: false` + LR: 5e-6
**How:** Followed the recipe in the system prompt
**Verdict:** Correct outcome, but Gemini executed your strategy, not its own

### Run 2 — Call 1 (no recipe)
**Decision:** `freeze_backbone: false`, LR unchanged at 2e-5
**How:** Independent reasoning — immediately identified the frozen backbone as the bottleneck
**Verdict:** Right call on the key lever; got lucky that EuroSAT forgave the high LR

### Run 3 — All 3 calls (principled context)
**Decision:** batch_size 16→32→64. Backbone never unfrozen.
**How:** Principles document emphasised catastrophic forgetting risks → Gemini treated unfreezing as too dangerous → used all 3 calls on safe incremental adjustments
**Verdict:** Worst result of all three runs. The context made Gemini risk-averse, not smarter.

### Run 4 — Gemini 3.1 Pro (vague prompt, larger model)
**Decision:** batch_size: 1 → weight_decay: 0 → (same again)
**How:** Identified gradient noise as a plateau escape mechanism — not wrong in general, wrong for this bottleneck
**Verdict:** Creative but misdiagnosed. Larger model ≠ better decision. Backbone never identified as the bottleneck.

### Run 5 — GPT-5 (vague prompt, direct OpenAI API)
**Decision:** `freeze_backbone: false`, LR unchanged at 2e-5
**How:** Same as Run 2 — immediately diagnosed frozen backbone as the ceiling
**Verdict:** One call, correct decision, early stop at F1=0.9757. Matches Gemini Flash's reasoning from a different model family.

---

## Total Cost Comparison

| Run | Model | Prompt Style | Best F1 | Calls | Tokens | Backbone unfrozen? |
|-----|-------|-------------|---------|-------|--------|--------------------|
| 1 | Gemini 2.5 Flash | Prescriptive | 0.9688 | 3 | ~2,630 | Yes (call 3) |
| 2 | Gemini 2.5 Flash | Vague | **0.9791** | **1** | **596** | Yes (call 1) |
| 3 | Gemini 2.5 Flash | Principled | 0.9128 | 3 | ~3,485 | **Never** |
| 4 | Gemini 3.1 Pro | Vague | 0.9360 | 3 | ~5,100 | Never |
| 5 | GPT-5 | Vague | 0.9757 | 1 | 2,863 | Yes (call 1) |

Human decisions across all five runs: **0**.

---

## MLflow: The Audit Trail

Every epoch logs:

```python
mlflow.log_metrics({
    "f1_macro":          metrics["f1_macro"],
    "train_loss":        train_loss,
    "learning_rate":     current_config["learning_rate"],
    "batch_size":        float(active_batch_size),
    "claude_tokens_used": float(tokens_this_epoch),   # 0 unless Gemini called
    "claude_suggested":  float(claude_suggested),      # 1.0 only at intervention epochs
}, step=epoch)
```

In the MLflow UI, the story is fully readable from metrics alone:

1. `f1_macro` climbs steadily, then flattens around 0.90
2. `claude_suggested` spikes at epochs 10, 11, 12
3. `learning_rate` drops to 5e-6 at epoch 13
4. `f1_macro` jumps at epoch 13

You can reconstruct every decision Gemini made and its effect on the model without reading a single line of code.

---

## The Code That Makes the Loop Tick

### The Config Reload (why it enables real-time adaptation)

```python
for epoch in range(1, num_epochs + 1):
    current_config = load_config(config_path)   # ← every epoch, fresh read
    
    # Detect if freeze_backbone changed (Gemini may have written it)
    if new_freeze != active_freeze:
        active_freeze = new_freeze
        apply_backbone_freeze(model, active_freeze)
        optimizer = build_optimizer(model, current_config["learning_rate"], ...)
```

No special handling for "Gemini wrote this." The loop doesn't know or care. It just reads the file.

### The Plateau Check and Advisor Call

```python
plateau_detected = is_plateau(f1_history, config["min_delta"], config["plateau_window"])

if plateau_detected and claude_call_count < config["max_claude_calls"]:
    tokens = request_config_update(current_config, epoch_history, epoch, config_path)
    claude_call_count += 1
    current_config = load_config(config_path)   # pick up Gemini's changes immediately
```

### The Advisor Call (full function)

```python
def request_config_update(config, epoch_history, current_epoch, config_path):
    payload = {
        "task": "image_classification",
        "model": config["model"],
        "dataset": config["dataset"],
        "current_epoch": current_epoch,
        "history": epoch_history,
        "freeze_backbone": config.get("freeze_backbone", False),
        "target_f1": config["target_f1"],
        "remaining_epochs": config["num_epochs"] - current_epoch,
    }

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        max_tokens=512,
        messages=[
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(payload)},
        ],
    )

    advisor_response = yaml.safe_load(response.choices[0].message.content)
    
    # Apply only adjustable fields — protected fields stay untouched
    final_config = dict(config)
    for field in ADJUSTABLE_FIELDS:
        if field in advisor_response:
            final_config[field] = advisor_response[field]

    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)

    return response.usage.total_tokens
```

---

## Key Lessons

### 1. The loop is the product
Your expertise no longer lives in the decisions you make during a run. It lives in the signals you choose to detect, the constraints you impose on the advisor, and the interface you design between the loop and the LLM.

### 2. Config-as-interface eliminates fragility
When the LLM writes a file rather than returning free text, there is nothing to parse. The format is the YAML schema. Deviations raise exceptions. The output is either valid or explicitly failed.

### 3. Budget the advisor calls deliberately
Calling the advisor every epoch is wasteful and noisy. Calling it only on plateau gives each call maximum signal. One call over 10 epochs cost less than a cent and produced a 7-point F1 gain.

### 4. Protected fields are the safety layer
Giving an LLM write access to a config sounds dangerous. It isn't, if you define the field whitelist carefully. The LLM can only adjust the levers you explicitly exposed. It cannot change the model, the dataset, or the evaluation target.

### 5. Backbone freezing is both a training technique and a loop engineering trick
Frozen backbone creates a natural plateau that forces the loop to exercise its adaptation mechanism. Progressive unfreezing — here done autonomously by Gemini — is the standard recipe for getting the most out of a pretrained model, and the loop executed it without being told to.

### 6. Context that emphasises risk makes LLMs risk-averse
Run 3 is the most important finding. The principles document described catastrophic forgetting so clearly that Gemini avoided unfreezing entirely — across all 3 calls — and scored 0.9128 vs 0.9791 with no context at all. More information produced worse decisions. The LLM optimised for safety over effectiveness because the context made risk salient.

### 7. Use context to correct specific failures, not to constrain strategy
The principles document was written to fix Run 2's side effects (zeroed weight_decay, high LR when unfreezing). But it was framed too broadly — it described the whole unfreezing risk, not just the paired-LR adjustment. A better document would have said: "when unfreezing, lower the LR proportionally" — targeted at the specific gap, not the whole decision.

### 8. LLMs have ML knowledge — test it before adding context
Run 2 proved Gemini knew about progressive unfreezing without being told. The prescriptive prompt in Run 1 made Gemini conservative by giving it a checklist. The principles document in Run 3 made it even more conservative by amplifying risk. The baseline — vague prompt, no context — produced the best result. Add context only where the LLM demonstrably fails, and scope it tightly to that failure.

### 9. Larger model ≠ better loop advisor
Gemini 3.1 Pro underperformed Gemini 2.5 Flash on the same prompt. Pro's reasoning was more elaborate — it found a genuinely interesting strategy in batch_size:1 — but elaborate reasoning doesn't mean correct diagnosis. A smaller, faster model that correctly identifies the bottleneck beats a larger one that finds the wrong lever. In a loop engineering context, correctness of the first call matters more than depth of reasoning.

### 10. Two models, same correct answer: the bottleneck is real
Gemini 2.5 Flash (Run 2) and GPT-5 (Run 5) both used the vague prompt, both made one call, and both immediately unfroze the backbone. Different model families, same diagnosis. This is evidence that the correct answer was deducible from the training history alone — and that both models had sufficient ML knowledge to find it.

### 11. "One change at a time" breaks for coupled parameters
The system prompt says "make one targeted adjustment per call." This is correct for independent parameters. But `freeze_backbone` and `learning_rate` are coupled — unfreezing is safest when LR is simultaneously reduced. The one-change rule made it impossible to do both in one call. The loop's recovery mechanism is the answer: if the high LR caused instability, the next plateau call would have reduced it. The system is designed for recovery, not perfection.

### 12. Production robustness matters as much as ML correctness
GPT-5 introduced two bugs that had nothing to do with ML: `max_tokens` vs `max_completion_tokens` (API parameter name change), and numeric values returned as quoted strings (`"2e-05"` instead of `2e-05`). Both crashed the training loop. A test suite that makes a real API call with a minimal payload catches these before a 40-minute training run does. The infrastructure around the LLM call is as important as the prompt inside it.

---

## How to Run This Yourself

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (or CPU for smoke test)
- OpenRouter API key at `openrouter.ai`

### Setup

```bash
git clone <repo>
cd loop

# Create venv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Add API key
echo "OPENROUTER_API_KEY=your_key_here" > .env
set -a && source .env && set +a
```

### Smoke test (CPU, 2 epochs, 10 samples, ~30 seconds)

```bash
python3 src/train.py --smoke-test
```

### Full training run

Reset config to starting values first:

```yaml
# config/image_eurosat.yaml
freeze_backbone: true
learning_rate: 2.0e-5
batch_size: 16
warmup_steps: 100
```

Then run:

```bash
python3 src/train.py
```

### View results

```bash
.venv/bin/mlflow ui --port 5000
# Open http://localhost:5000
```

---

## What Comes Next

The fine-tuned ViT backbone (best checkpoint saved in MLflow) is the feature extractor for a satellite change detection pipeline — comparing feature representations of the same location at two different timestamps to detect land cover change. The loop engineering pattern carries over unchanged: same plateau detection, same advisor pattern, different metric (change detection F1 rather than classification F1).

The lesson scales: any training loop with a measurable target can be turned into a loop that adjusts itself. The components — detect signal, call advisor, write config, read config — are reusable across any domain.

---

*Full source: `src/` directory. Design documentation: `LOOP_ENGINEERING.md`. MLflow experiment: `loop_engineering_v1`.*
