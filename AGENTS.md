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
- Because the current `one2one` branch is still anchor-dense, NMS-free inference relies on two mechanisms that must be preserved together:
  - unique assignment during training
  - local peak filtering during inference
- The score used at inference is a quality-aware prompt-conditioned score, not a pure objectness score and not a pure class score.
- PromptDET currently works best when prompt matching uses both:
  - category-level global prototypes
  - category-level detail / token memory
  Do not collapse matching back to a single local-token `max` path or a closed-set classifier style head.

## Prompt Conditioning Rules
- Prompt classes are represented by dynamically aggregated visual prototypes.
- When multiple prompt instances belong to the same category, their features should be fused first to form a more stable category-level prompt representation.
- Different prompt categories should not be forced to compete too early inside a shared prompt-conditioning path. Prefer category-wise conditional branches or other designs that keep cross-category interference low until late aggregation.
- Prompt slot indices are randomized per episode. This is intentional and must be preserved.
- Do not add logic that lets the model memorize fixed dataset `category_id -> slot` mappings.
- Any new feature must continue to work when prompt classes appear:
  - in one image or many images
  - with one instance or multiple instances
  - as same-class, partially overlapping, or mixed-class prompt sets

## Sampling and Training Constraints
- Remove `same_instance`-specific sampling rules. Training should use a single class-conditioned prompt-set regime.
- In each training episode, sample `min_prompt_classes` to `max_prompt_classes` categories to form the prompt set.
- Prompt instances may come from one image or multiple images. A class may appear once or multiple times across the prompt set.
- Query sampling should stay approximately balanced between positive and negative episodes by design, rather than relying on uncontrolled random draws.
- Positive query images must contain one or more prompt-defined classes, but they do not need to contain every prompt-defined class.
- When multiple positive query candidates exist, prefer images that:
  - cover more of the prompt-defined classes
  - contain more instances of those prompt-defined classes
  - avoid excessive unrelated objects when possible
- Negative query images must be chosen from labeled images that contain no prompt-defined classes.
- During training, when the prompt set contains multiple categories:
  - same-category prompt instances should be fused into a category-level representation
  - different categories should be processed with separate category-conditional branches on the query side as much as practical
- The training objective should teach the model both:
  - prompt-conditioned detection of the prompt-defined categories
  - empty output when the current query contains none of the prompt-defined categories
- The purpose of this training regime is twofold:
  - learn prompt-conditioned category detection
  - learn empty-category rejection for the current prompt set
- If adding new losses or assigners, preserve the distinction:
  - `one2many` improves optimization
  - `one2one` determines final prediction behavior

## Detection Head and Loss Guidelines
- Keep objectness and prompt-classification separated conceptually.
- Background rejection should not rely only on threshold tuning.
- Box regression should be supervised with quality-aware matching and center-aware assignment.
- If duplicates reappear, fix assignment or one-to-one supervision first, not post-processing first.
- The `one2one` branch should explicitly suppress non-matched neighbors around each matched target. Duplicate suppression belongs in supervision, not in NMS.
- If class confusion persists, prefer improvements to:
  - prompt prototype quality
  - contrastive separation
  - class margin / calibration
  instead of adding closed-set shortcuts.
- Negative supervision for prompt-conditioned scores must suppress all active prompt classes on non-target regions, not only the current max-scoring class.
- If `joint_score` or related quality-aware scoring losses are modified, keep them aligned with inference scoring semantics. Do not train one notion of quality and decode with another.
- Context / grounding priors may assist ranking, but they must remain auxiliary. Do not let slot priors become the sole source of class evidence.

## Inference Rules
- Detection takes a prompt set built from one or more labeled images.
- Each prompt image may contribute one or more selected categories as targets.
- Selected categories across prompt images may be identical, partially overlapping, or completely different.
- The same category may appear in multiple prompt images with multiple prompt instances.
- At inference time, when multiple prompt instances belong to the same category, fuse them into a single, more stable category representation before matching on the query.
- Different prompt categories should then be matched on the query through separate category-conditional branches or equivalent low-interference logic, rather than a single early shared competition step.
- The model must dynamically infer the active target set from all selected prompt annotations and only return query detections that belong to those prompt-defined classes.
- If a prompt-defined category is absent from the query image, that category's output should be empty.
- The main inference path should be:
  - decode `one2one`
  - build final prompt-conditioned per-class scores
  - enforce one-anchor-to-one-prompt-class ownership
  - apply local peak filtering on those final class-conditional scores
  - threshold
  - top-k
- Avoid category-wise suppression as a substitute for correct one-to-one learning.
- NMS is not part of the intended final architecture.
- Do not use shared `objectness * targetness` peak filtering as the main candidate selector before prompt-class assignment. That design was empirically shown to leak non-prompt objects into active classes.
- Local peak filtering must operate after prompt-conditioned class ownership is decided, not before.
- Keep prompt-vs-null gating and class-margin gating in the final inference score. Prompt-conditioned rejection must be part of the main score, not a post-hoc heuristic.

## Data and Toy Dataset
- `toy_data` labels must stay consistent with the generated `train.txt`, `val.txt`, `classes.txt`, and `prompt_set.json`.
- If modifying toy data generation, also update its consistency checks.
- Example prompt specs must be derived from real label files, never hand-waved placeholders.

## Validated Training Strategy
- Positive query sampling must not overfit to only clean prompt-target scenes. Hard positives with non-prompt distractor objects are required.
- Keep a controllable hard-positive sampling path that deliberately includes positive query images containing unrelated labeled objects, so the model learns `detect prompted targets` and `reject non-prompt objects in positive scenes` simultaneously.
- Negative episodes remain important for empty-set rejection, but they are not sufficient to solve the dominant precision failure mode. Do not rely on negative episodes alone to teach non-prompt rejection.
- When ranking positive query candidates, coverage of prompt classes and number of prompt instances still matter, but training must also expose the model to cluttered mixed-object scenes.

## Error Analysis Requirements
- Precision analysis must explicitly separate:
  - false positives on labeled non-prompt objects
  - false positives near true prompt targets
  - false positives on background
- Do not rely on aggregate precision or `mean_bg_score` alone when diagnosing failures. A low mean background score can still hide severe false positives on a small number of non-prompt objects.
- Treat `false_positives_on_non_target_object` as a first-class validation signal. This was the dominant failure mode across earlier experiments.

## Known Failure Modes To Avoid
- Do not regress to a design where any salient object center can be promoted first and only weakly filtered by prompt classes afterward.
- Do not let context priors paint unrelated objects as active prompt slots without strong prototype evidence.
- Do not assume removing duplicate predictions is enough. Earlier iterations showed that cross-class duplicates can be reduced while single-class non-prompt false positives remain the main bottleneck.
- Do not interpret improvements in recall or F1 alone as success if `false_positives_on_non_target_object` remains high.

## Engineering Expectations
- Favor changes that improve identifiability, calibration, and prompt-conditioned generalization over cosmetic fixes.
- Do not revert to a fixed-class detector formulation.
- When changing model outputs, update:
  - training loss
  - evaluator
  - detect entrypoint
  - config defaults
  - documentation/examples
- When changing prompt-matching, inference scoring, sampling, or evaluator metrics, update this file if the change becomes part of the intended long-term design rather than a temporary ablation.
