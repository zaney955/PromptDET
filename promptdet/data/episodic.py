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
        negative_ratio: float = 0.35,
        hard_negative_ratio: float = 0.35,
        prompt_type: str = "same_category",
        min_prompt_classes: int = 1,
        max_prompt_classes: int = 3,
        max_prompt_instances_per_class: int = 2,
        max_prompt_images: int = 4,
        confusable_non_target_weight: float = 2.0,
    ):
        super().__init__()
        payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.episodes_per_epoch = episodes_per_epoch
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.prompt_type = prompt_type
        self.min_prompt_classes = min_prompt_classes
        self.max_prompt_classes = max_prompt_classes
        self.max_prompt_instances_per_class = max_prompt_instances_per_class
        self.max_prompt_images = max_prompt_images
        self.confusable_non_target_weight = confusable_non_target_weight

        self.image_records = {item["id"]: item for item in payload["images"]}
        self.image_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.image_to_class_ids: Dict[int, set[int]] = defaultdict(set)
        self.class_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.class_to_image_ids: Dict[int, set[int]] = defaultdict(set)
        self.class_image_to_anns: Dict[int, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
        self.categories = {item["id"]: item["name"] for item in payload["categories"]}
        self.class_to_family: Dict[int, str] = {}
        self.family_to_classes: Dict[str, set[int]] = defaultdict(set)
        for category_id, category_name in self.categories.items():
            family = category_name.split("_")[-1] if "_" in category_name else category_name
            self.class_to_family[category_id] = family
            self.family_to_classes[family].add(category_id)
        for ann in payload["annotations"]:
            bbox = ann["bbox"]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            self.image_to_anns[image_id].append(ann)
            self.image_to_class_ids[image_id].add(category_id)
            self.class_to_anns[category_id].append(ann)
            self.class_to_image_ids[category_id].add(image_id)
            self.class_image_to_anns[category_id][image_id].append(ann)

        self.image_ids = sorted(self.image_records.keys())
        self.classes = sorted(self.class_to_anns.keys())
        self.non_empty_image_ids = [image_id for image_id in self.image_ids if self.image_to_anns[image_id]]

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _load_image(self, image_id: int) -> Image.Image:
        path = self.images_dir / self.image_records[image_id]["file_name"]
        return Image.open(path).convert("RGB")

    def _sample_prompt_classes(self) -> List[int]:
        max_classes = min(self.max_prompt_classes, len(self.classes))
        min_classes = min(self.min_prompt_classes, max_classes)
        count = random.randint(min_classes, max_classes)
        return random.sample(self.classes, count)

    def _sample_prompt_instances(self, prompt_class_ids: List[int], class_to_slot: Dict[int, int]) -> List[dict]:
        selected_instances: List[dict] = []
        used_image_ids: set[int] = set()
        used_ann_ids: set[int] = set()

        for class_id in prompt_class_ids:
            class_slot = class_to_slot[class_id]
            image_ids = list(self.class_to_image_ids[class_id])
            random.shuffle(image_ids)
            max_instances = min(
                self.max_prompt_instances_per_class,
                max(len(self.class_to_anns[class_id]), 1),
            )
            target_instances = random.randint(1, max_instances)
            class_instances: List[dict] = []

            for image_id in image_ids:
                if len(class_instances) >= target_instances:
                    break
                if len(used_image_ids) >= self.max_prompt_images and image_id not in used_image_ids:
                    continue
                candidates = [ann for ann in self.class_image_to_anns[class_id][image_id] if ann["id"] not in used_ann_ids]
                if not candidates:
                    candidates = self.class_image_to_anns[class_id][image_id]
                ann = random.choice(candidates)
                used_image_ids.add(image_id)
                used_ann_ids.add(ann["id"])
                class_instances.append({
                    "image_id": image_id,
                    "bbox": ann["bbox"],
                    "class_slot": class_slot,
                    "category_id": class_id,
                })

            if len(class_instances) < target_instances:
                fallback_images = list(self.class_to_image_ids[class_id])
                random.shuffle(fallback_images)
                for image_id in fallback_images:
                    if len(class_instances) >= target_instances:
                        break
                    candidates = self.class_image_to_anns[class_id][image_id]
                    ann = random.choice(candidates)
                    class_instances.append({
                        "image_id": image_id,
                        "bbox": ann["bbox"],
                        "class_slot": class_slot,
                        "category_id": class_id,
                    })

            selected_instances.extend(class_instances)
        return selected_instances

    def _get_confusable_classes(self, prompt_class_ids: List[int]) -> set[int]:
        prompt_class_set = set(prompt_class_ids)
        confusable = set()
        for class_id in prompt_class_ids:
            family = self.class_to_family.get(class_id)
            if family is None:
                continue
            confusable.update(self.family_to_classes[family] - prompt_class_set)
        return confusable

    def _sample_query(self, prompt_class_ids: List[int], prompt_image_ids: List[int]) -> Tuple[int, bool]:
        prompt_class_set = set(prompt_class_ids)
        prompt_image_set = set(prompt_image_ids)
        confusable_class_set = self._get_confusable_classes(prompt_class_ids)
        positive_ids = sorted(set().union(*(self.class_to_image_ids[class_id] for class_id in prompt_class_ids)))
        positive_ids = [image_id for image_id in positive_ids if image_id not in prompt_image_set] or positive_ids
        confusable_positive_ids = [
            image_id
            for image_id in positive_ids
            if self.image_to_class_ids[image_id] & confusable_class_set
        ]

        negative_ids = [
            image_id
            for image_id in self.image_ids
            if not (self.image_to_class_ids[image_id] & prompt_class_set) and image_id not in prompt_image_set
        ]
        confusable_negative_ids = [
            image_id
            for image_id in negative_ids
            if self.image_to_class_ids[image_id] & confusable_class_set
        ]
        hard_negative_ids = [
            image_id
            for image_id in negative_ids
            if len(self.image_to_class_ids[image_id]) > 0
        ]

        sample = random.random()
        if sample < self.hard_negative_ratio:
            if confusable_negative_ids:
                return random.choice(confusable_negative_ids), False
            if hard_negative_ids:
                return random.choice(hard_negative_ids), False
        if sample < self.hard_negative_ratio + self.negative_ratio and negative_ids:
            return random.choice(negative_ids), False
        if positive_ids:
            if confusable_positive_ids and random.random() < 0.5:
                return random.choice(confusable_positive_ids), True
            return random.choice(positive_ids), True
        if confusable_negative_ids:
            return random.choice(confusable_negative_ids), False
        if hard_negative_ids:
            return random.choice(hard_negative_ids), False
        if negative_ids:
            return random.choice(negative_ids), False
        return random.choice(self.image_ids), False

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sampled_prompt_class_ids = self._sample_prompt_classes()
        slot_order = random.sample(range(len(sampled_prompt_class_ids)), len(sampled_prompt_class_ids))
        class_to_slot = {
            class_id: slot_order[idx]
            for idx, class_id in enumerate(sampled_prompt_class_ids)
        }
        prompt_class_ids = [-1] * len(sampled_prompt_class_ids)
        for class_id, slot in class_to_slot.items():
            prompt_class_ids[slot] = class_id

        prompt_instances = self._sample_prompt_instances(sampled_prompt_class_ids, class_to_slot)
        prompt_image_ids = [instance["image_id"] for instance in prompt_instances]
        query_image_id, positive = self._sample_query(sampled_prompt_class_ids, prompt_image_ids)

        prompt_images = []
        prompt_boxes = []
        prompt_class_indices = []
        for instance in prompt_instances:
            image = self._load_image(instance["image_id"])
            box = torch.tensor([instance["bbox"]], dtype=torch.float32)
            image_tensor, box = resize_image_and_boxes(image, box, self.image_size)
            prompt_images.append(image_tensor)
            prompt_boxes.append(box[0])
            prompt_class_indices.append(instance["class_slot"])

        query_image = self._load_image(query_image_id)
        query_anns = self.image_to_anns[query_image_id]
        query_boxes = []
        query_labels = []
        query_category_ids = []
        non_target_boxes = []
        non_target_weights = []
        confusable_class_set = self._get_confusable_classes(sampled_prompt_class_ids)
        for ann in query_anns:
            class_slot = class_to_slot.get(ann["category_id"])
            if class_slot is None:
                non_target_boxes.append(ann["bbox"])
                non_target_weights.append(
                    self.confusable_non_target_weight if ann["category_id"] in confusable_class_set else 1.0
                )
                continue
            query_boxes.append(ann["bbox"])
            query_labels.append(class_slot)
            query_category_ids.append(ann["category_id"])

        if query_boxes:
            query_boxes_tensor = torch.tensor(query_boxes, dtype=torch.float32)
            query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)
            query_category_ids_tensor = torch.tensor(query_category_ids, dtype=torch.long)
        else:
            query_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            query_labels_tensor = torch.zeros((0,), dtype=torch.long)
            query_category_ids_tensor = torch.zeros((0,), dtype=torch.long)

        if non_target_boxes:
            non_target_boxes_tensor = torch.tensor(non_target_boxes, dtype=torch.float32)
            non_target_weights_tensor = torch.tensor(non_target_weights, dtype=torch.float32)
        else:
            non_target_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            non_target_weights_tensor = torch.zeros((0,), dtype=torch.float32)

        query_image_tensor, query_boxes_tensor = resize_image_and_boxes(query_image, query_boxes_tensor, self.image_size)
        _, non_target_boxes_tensor = resize_image_and_boxes(query_image, non_target_boxes_tensor, self.image_size)

        return {
            "prompt_images": torch.stack(prompt_images, dim=0),
            "prompt_boxes": torch.stack(prompt_boxes, dim=0),
            "prompt_class_indices": torch.tensor(prompt_class_indices, dtype=torch.long),
            "prompt_class_ids": torch.tensor(prompt_class_ids, dtype=torch.long),
            "prompt_type": torch.tensor(0, dtype=torch.long),
            "query_image": query_image_tensor,
            "query_boxes": query_boxes_tensor,
            "query_labels": query_labels_tensor,
            "query_category_ids": query_category_ids_tensor,
            "query_non_target_boxes": non_target_boxes_tensor,
            "query_non_target_weights": non_target_weights_tensor,
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
            "is_positive": torch.tensor(int(positive and query_boxes_tensor.shape[0] > 0), dtype=torch.long),
        }


def collate_episodes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
    batch_size = len(batch)
    max_prompt_instances = max(item["prompt_images"].shape[0] for item in batch)
    max_prompt_classes = max(item["prompt_class_ids"].shape[0] for item in batch)
    channels, height, width = batch[0]["query_image"].shape

    prompt_images = torch.zeros((batch_size, max_prompt_instances, channels, height, width), dtype=torch.float32)
    prompt_boxes = torch.zeros((batch_size, max_prompt_instances, 4), dtype=torch.float32)
    prompt_class_indices = torch.zeros((batch_size, max_prompt_instances), dtype=torch.long)
    prompt_instance_mask = torch.zeros((batch_size, max_prompt_instances), dtype=torch.bool)
    prompt_class_ids = torch.full((batch_size, max_prompt_classes), -1, dtype=torch.long)
    prompt_class_mask = torch.zeros((batch_size, max_prompt_classes), dtype=torch.bool)

    for batch_idx, item in enumerate(batch):
        num_instances = item["prompt_images"].shape[0]
        num_classes = item["prompt_class_ids"].shape[0]
        prompt_images[batch_idx, :num_instances] = item["prompt_images"]
        prompt_boxes[batch_idx, :num_instances] = item["prompt_boxes"]
        prompt_class_indices[batch_idx, :num_instances] = item["prompt_class_indices"]
        prompt_instance_mask[batch_idx, :num_instances] = True
        prompt_class_ids[batch_idx, :num_classes] = item["prompt_class_ids"]
        prompt_class_mask[batch_idx, :num_classes] = True

    return {
        "prompt_images": prompt_images,
        "prompt_boxes": prompt_boxes,
        "prompt_class_indices": prompt_class_indices,
        "prompt_instance_mask": prompt_instance_mask,
        "prompt_class_ids": prompt_class_ids,
        "prompt_class_mask": prompt_class_mask,
        "prompt_type": torch.stack([item["prompt_type"] for item in batch], dim=0),
        "query_image": torch.stack([item["query_image"] for item in batch], dim=0),
        "targets": [
            {
                "boxes": item["query_boxes"],
                "labels": item["query_labels"],
                "category_ids": item["query_category_ids"],
                "non_target_boxes": item["query_non_target_boxes"],
                "non_target_weights": item["query_non_target_weights"],
                "image_size": int(item["image_size"].item()),
            }
            for item in batch
        ],
        "is_positive": torch.stack([item["is_positive"] for item in batch], dim=0),
    }
