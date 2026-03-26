# PromptDET

PromptDET is a prompt-conditioned detector built from scratch for the workflow:

- prompt set of one or more annotated images
- one or more selected categories defined by prompt examples
- query image

The model predicts all query boxes that belong to the categories dynamically defined by the prompt set.
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

- `./toy_data/train.json`
- `./toy_data/val.json`
- `./toy_data/images/*.png`

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

You can override any important path from the CLI:

```bash
python train.py \
  --config configs/toy_train.json \
  --train-annotations ./toy_data/train.json \
  --val-annotations ./toy_data/val.json \
  --images-dir ./toy_data/images \
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
  --prompt-box 62 57 125 120 \
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

`prompt_spec` format:

```json
{
  "prompts": [
    {
      "image": "./images/train_00000.png",
      "annotations": [
        {"bbox": [62, 57, 125, 120], "label": 0},
        {"bbox": [173, 189, 214, 230], "label": 2}
      ]
    },
    {
      "image": "./images/train_00007.png",
      "annotations": [
        {"bbox": [68, 87, 164, 183], "label": 1}
      ]
    }
  ]
}
```

When `prompt_set.json` lives inside `toy_data/`, image paths are resolved relative to that JSON file. The `label` field is the dataset `category_id` from `train.json`, not a free-form class name.
For the toy dataset, this keeps `prompt_set.json` consistent with `train.json`.
Conceptually, `label` is just the identity token used to indicate which prompt annotations belong to the same prompt-defined class within that episode.

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
- combine objectness and prompt-class confidence into a quality-aware score
- threshold
- top-k

The main inference path is NMS-free by design. `nms_iou_threshold` may still exist in configs or CLI for compatibility, but NMS is not part of the intended final prediction behavior.

## Current Scope

Implemented:

- full modular prompt-conditioned detector with prompt-set episodes
- shared backbone and PAN/FPN neck
- prompt crop encoder with multi-instance class prototype aggregation
- cross-attention prompt/query fusion
- dual-branch dense detection head with DFL regression
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

The project expects a COCO-like JSON with:

```json
{
  "images": [{"id": 0, "file_name": "xxx.png", "width": 256, "height": 256}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [x1, y1, x2, y2]}],
  "categories": [{"id": 1, "name": "class_name"}]
}
```

Bounding boxes use absolute `xyxy` pixel coordinates.
