# PromptDET

PromptDET is a prompt-conditioned detector built for the workflow:

- prompt set of one or more annotated images
- one or more selected categories defined by prompt examples
- query image

The model predicts only the query boxes that belong to the categories dynamically defined by the prompt set.
Prompt labels are episode-local identity tokens used to group prompt instances into classes for that episode. They are not natural-language semantics and should not be treated as globally meaningful category embeddings.

The implementation combines:

- SegGPT-style prompt-conditioned support/query interaction
- YOLO-style multi-scale dense detection
- DFL-based box regression
- dual detection heads:
  - `one2many` for dense auxiliary supervision during training
  - `one2one` for unique-matching final inference
- quality-aware matching and NMS-free `one2one` inference with local peak filtering

## Project Layout

```text
PromptDET/
  promptdet/
    config.py
    data/
    engine/
    models/
    utils/
  configs/
  scripts/
  train.py
  detect.py
```

## Install

```bash
pip install -r requirements.txt
```

## Generate Toy Data

```bash
python scripts/make_toy_dataset.py --output-dir ./toy_data
```

This creates:

- `./toy_data/train.txt`
- `./toy_data/val.txt`
- `./toy_data/classes.txt`
- `./toy_data/dataset.yaml`
- `./toy_data/images/*.png`
- `./toy_data/labels/train/*.txt`
- `./toy_data/labels/val/*.txt`
- `./toy_data/prompt_set.json`

## Train

```bash
python train.py --config configs/toy_train.json --device cpu
```

Single-node multi-GPU training with DDP:

```bash
torchrun --nproc_per_node=4 train.py --config configs/toy_train.json --device cuda
```

To suppress the `OMP_NUM_THREADS` warning from `torchrun`, set it explicitly:

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 train.py --config configs/toy_train.json --device cuda
```

You can override the dataset paths from the CLI:

```bash
python train.py \
  --config configs/toy_train.json \
  --train-list ./toy_data/train.txt \
  --val-list ./toy_data/val.txt \
  --train-labels-dir ./toy_data/labels/train \
  --val-labels-dir ./toy_data/labels/val \
  --class-names ./toy_data/classes.txt \
  --output-dir ./outputs/exp1 \
  --device cuda
```

The `batch_size` argument is the per-process batch size. For example, `torchrun --nproc_per_node=4 ... --batch-size 8` gives an effective global batch size of `32`.

Sampling behavior is controlled in `configs/toy_train.json`:

- `data.min_prompt_classes` / `data.max_prompt_classes`
- `data.max_prompt_instances_per_class`
- `data.max_prompt_images`
- `data.negative_ratio`
- `data.hard_negative_ratio`

## Inference

Single-prompt compatibility mode:

```bash
python detect.py \
  --config ./outputs/exp1/config.json \
  --checkpoint ./outputs/exp1/best.pt \
  --prompt-image ./toy_data/images/train_00000.png \
  --prompt-box 0.365234 0.345703 0.246094 0.246094 \
  --prompt-label 0 \
  --query-image ./toy_data/images/val_00001.png \
  --output-dir ./outputs/infer_demo \
  --conf-threshold 0.15
```

Prompt-set mode:

```bash
python detect.py \
  --config ./outputs/exp1/config.json \
  --checkpoint ./outputs/exp1/best.pt \
  --prompt-spec ./toy_data/prompt_set.json \
  --query-image ./toy_data/images/val_00001.png \
  --output-dir ./outputs/infer_prompt_set \
  --conf-threshold 0.15
```

Batch directory mode:

```bash
python detect.py \
  --config ./outputs/exp1/config.json \
  --checkpoint ./outputs/exp1/best.pt \
  --prompt-spec ./toy_data/prompt_set.json \
  --query-image ./toy_data/images \
  --output-dir ./outputs/infer_batch \
  --conf-threshold 0.15
```

When `--query-image` points to a directory, each image is written to its own subdirectory under `--output-dir`, and a `batch_summary.json` file is also produced.

`prompt_spec` format:

```json
{
  "prompts": [
    {
      "image": "./images/train_00000.png",
      "annotations": [
        {"bbox": [0.365234, 0.345703, 0.246094, 0.246094], "label": 0},
        {"bbox": [0.755859, 0.818359, 0.160156, 0.160156], "label": 2}
      ]
    },
    {
      "image": "./images/train_00007.png",
      "annotations": [
        {"bbox": [0.453125, 0.527344, 0.375000, 0.375000], "label": 1}
      ]
    }
  ]
}
```

When `prompt_set.json` lives inside `toy_data/`, image paths are resolved relative to that JSON file. The `label` field uses the class id defined by line order in `classes.txt`, not a free-form class name. Conceptually, `label` is just the identity token used to indicate which prompt annotations belong to the same prompt-defined class within that episode.

You can validate the generated toy dataset and prompt spec with:

```bash
python scripts/check_toy_data.py --data-dir ./toy_data
```

Outputs:

- `prediction.json`
- `prediction.png`

## Detection Behavior

Inference uses the `one2one` branch only.
The intended prediction path is:

- decode `one2one`
- apply local peak filtering
- combine objectness, prompt-conditioned targetness, and prompt-class confidence into a quality-aware score
- threshold
- top-k

The main inference path is NMS-free by design. `nms_iou_threshold` may still exist in configs or CLI for compatibility, but NMS is not part of the intended final prediction behavior.

## Current Scope

Implemented:

- full modular prompt-conditioned detector with prompt-set episodes
- shared backbone and PAN/FPN neck
- prompt crop encoder with multi-instance class prototype aggregation
- cross-attention prompt/query fusion with null-aware gating
- dual-branch dense detection head with DFL regression
- prompt-conditioned targetness prediction
- dynamic prompt-class assignment
- `one2one` unique matching with duplicate suppression supervision
- local peak filtering for NMS-free `one2one` inference
- training and validation loops
- single-node distributed training with DDP
- standalone inference script
- toy dataset generator for smoke tests

Not yet implemented:

- multi-prompt inference pool
- same-instance mode
- text prompt encoder

## Dataset Format

The project now uses a YOLO-style dataset layout:

- `train.txt` / `val.txt`: one image path per line
- `labels/<split>/<stem>.txt`: one object per line in `class x_center y_center width height`
- `classes.txt`: one class name per line
- `prompt_set.json`: prompt annotations using the same normalized `xywh` format

All four box values are normalized to `0~1` relative to image width and height.
