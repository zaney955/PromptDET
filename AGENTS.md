# PromptDET Development Notes

## Core Task Definition
- PromptDET is a prompt-conditioned detector, not a closed-set detector.
- A prompt set dynamically defines the target classes for each episode.
- Prompt labels are identity tokens within the episode. They are not natural-language semantics and must not be treated as globally meaningful category embeddings.
- Query outputs must include only objects belonging to the prompt-defined classes. Any other object is background.

## Current Architecture
- The model uses a dual-head detection design:
  - `one2many`: dense auxiliary training branch for optimization stability and recall.
  - `one2one`: unique-matching branch for final inference.
- Inference must use the `one2one` branch only.
- Inference is NMS-free by design. Do not reintroduce NMS into the main prediction path unless explicitly requested for ablation.
- The score used at inference is a quality-aware combination of objectness and prompt-class confidence.

## Prompt Conditioning Rules
- Prompt classes are represented by dynamically aggregated visual prototypes.
- Prompt slot indices are randomized per episode. This is intentional and must be preserved.
- Do not add logic that lets the model memorize fixed dataset `category_id -> slot` mappings.
- Any new feature must continue to work when prompt classes appear:
  - in one image or many images
  - with one instance or multiple instances
  - as same-class or mixed-class prompt sets

## Sampling and Training Constraints
- Hard negatives should preferentially include confusable non-target objects, especially same-family shapes/colors.
- Sampling should continue to expose:
  - positive episodes
  - negative episodes
  - hard negative episodes
  - mixed prompt-class episodes
- If adding new losses or assigners, preserve the distinction:
  - `one2many` improves optimization
  - `one2one` determines final prediction behavior

## Detection Head and Loss Guidelines
- Keep objectness and prompt-classification separated conceptually.
- Background rejection should not rely only on threshold tuning.
- Box regression should be supervised with quality-aware matching and center-aware assignment.
- If duplicates reappear, fix assignment or one-to-one supervision first, not post-processing first.
- If class confusion persists, prefer improvements to:
  - prompt prototype quality
  - contrastive separation
  - class margin / calibration
  instead of adding closed-set shortcuts.

## Inference Rules
- The main inference path should be:
  - decode `one2one`
  - score
  - threshold
  - top-k
- Avoid category-wise suppression as a substitute for correct one-to-one learning.
- `nms_iou_threshold` may exist for compatibility, but NMS is not part of the intended final architecture.

## Data and Toy Dataset
- `toy_data` labels must stay consistent with the generated `train.json` and `prompt_set.json`.
- If modifying toy data generation, also update its consistency checks.
- Example prompt specs must be derived from real annotations, never hand-waved placeholders.

## Engineering Expectations
- Favor changes that improve identifiability, calibration, and prompt-conditioned generalization over cosmetic fixes.
- Do not revert to a fixed-class detector formulation.
- When changing model outputs, update:
  - training loss
  - evaluator
  - detect entrypoint
  - config defaults
  - documentation/examples
