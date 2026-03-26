from __future__ import annotations

from collections import defaultdict
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def resize_image_and_boxes(image: Image.Image, boxes: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_w, orig_h = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    boxes = boxes.clone().float()
    if boxes.numel() > 0:
        boxes[:, 0::2] *= size / orig_w
        boxes[:, 1::2] *= size / orig_h
    return _pil_to_tensor(resized), boxes


class PromptEpisodeDataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        image_size: int,
        episodes_per_epoch: int,
        negative_ratio: float = 0.25,
        hard_negative_ratio: float = 0.25,
        prompt_type: str = "same_category",
    ):
        super().__init__()
        payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.episodes_per_epoch = episodes_per_epoch
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.prompt_type = prompt_type

        self.image_records = {item["id"]: item for item in payload["images"]}
        self.image_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.class_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.class_to_image_ids: Dict[int, set[int]] = defaultdict(set)
        self.categories = {item["id"]: item["name"] for item in payload["categories"]}
        for ann in payload["annotations"]:
            bbox = ann["bbox"]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            self.image_to_anns[ann["image_id"]].append(ann)
            self.class_to_anns[ann["category_id"]].append(ann)
            self.class_to_image_ids[ann["category_id"]].add(ann["image_id"])

        self.image_ids = sorted(self.image_records.keys())
        self.classes = sorted(self.class_to_anns.keys())

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _load_image(self, image_id: int) -> Image.Image:
        path = self.images_dir / self.image_records[image_id]["file_name"]
        return Image.open(path).convert("RGB")

    def _sample_query(self, class_id: int) -> Tuple[int, bool]:
        r = random.random()
        positive_ids = list(self.class_to_image_ids[class_id])
        if r > self.negative_ratio + self.hard_negative_ratio or len(self.image_ids) == len(positive_ids):
            return random.choice(positive_ids), True

        negative_ids = [image_id for image_id in self.image_ids if image_id not in self.class_to_image_ids[class_id]]
        if not negative_ids:
            return random.choice(positive_ids), True
        return random.choice(negative_ids), False

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        class_id = random.choice(self.classes)
        support_ann = random.choice(self.class_to_anns[class_id])
        query_image_id, positive = self._sample_query(class_id)

        support_image = self._load_image(support_ann["image_id"])
        query_image = self._load_image(query_image_id)

        support_box = torch.tensor([support_ann["bbox"]], dtype=torch.float32)
        support_image_tensor, support_box = resize_image_and_boxes(support_image, support_box, self.image_size)

        query_anns = self.image_to_anns[query_image_id]
        query_boxes = []
        for ann in query_anns:
            if ann["category_id"] == class_id:
                query_boxes.append(ann["bbox"])
        if query_boxes:
            query_boxes = torch.tensor(query_boxes, dtype=torch.float32)
        else:
            query_boxes = torch.zeros((0, 4), dtype=torch.float32)
        query_image_tensor, query_boxes = resize_image_and_boxes(query_image, query_boxes, self.image_size)

        return {
            "support_image": support_image_tensor,
            "support_box": support_box[0],
            "prompt_label": torch.tensor(class_id, dtype=torch.long),
            "prompt_type": torch.tensor(0, dtype=torch.long),
            "query_image": query_image_tensor,
            "query_boxes": query_boxes,
            "query_labels": torch.full((query_boxes.shape[0],), class_id, dtype=torch.long),
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
            "is_positive": torch.tensor(int(positive), dtype=torch.long),
        }


def collate_episodes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
    return {
        "support_image": torch.stack([item["support_image"] for item in batch], dim=0),
        "support_box": torch.stack([item["support_box"] for item in batch], dim=0),
        "prompt_label": torch.stack([item["prompt_label"] for item in batch], dim=0),
        "prompt_type": torch.stack([item["prompt_type"] for item in batch], dim=0),
        "query_image": torch.stack([item["query_image"] for item in batch], dim=0),
        "targets": [
            {
                "boxes": item["query_boxes"],
                "labels": item["query_labels"],
                "image_size": int(item["image_size"].item()),
            }
            for item in batch
        ],
        "is_positive": torch.stack([item["is_positive"] for item in batch], dim=0),
    }
