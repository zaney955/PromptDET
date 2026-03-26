# PromptDET

PromptDET is a prompt-conditioned detector built from scratch for the workflow:

- prompt image
- prompt bounding box
- prompt label
- query image

The model predicts all query boxes that match the prompted target category. The implementation combines:

- SegGPT-style prompt-conditioned support/query interaction
- YOLO-style multi-scale dense detection
- DFL-based box regression
- task-aligned top-k assignment

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

## Inference

```bash
python detect.py \
  --config ./outputs/exp1/config.json \
  --checkpoint ./outputs/exp1/best.pt \
  --prompt-image ./toy_data/images/train_00000.png \
  --prompt-box 10 20 80 90 \
  --prompt-label 0 \
  --query-image ./toy_data/images/val_00001.png \
  --output-dir ./outputs/infer_demo
```

Outputs:

- `prediction.json`
- `prediction.png`

## Current Scope

Implemented:

- full modular prompt-conditioned detector
- shared backbone and PAN/FPN neck
- prompt crop encoder
- cross-attention prompt/query fusion
- dense detection head with DFL regression
- task-aligned assignment
- training and validation loops
- standalone inference script
- toy dataset generator for smoke tests

Not yet implemented:

- distributed training
- multi-prompt inference pool
- same-instance mode
- text prompt encoder
- end-to-end one-to-one NMS-free branch

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
