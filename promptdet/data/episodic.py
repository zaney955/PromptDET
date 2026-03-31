from __future__ import annotations

from collections import defaultdict
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from promptdet.data.prompt_hints import (
    build_prompt_hint_map,
    build_prompt_target_map,
    build_query_detection_targets,
    sample_slot_colors,
)
from promptdet.data.yolo_io import load_class_names, load_image_list, parse_yolo_label_file
from promptdet.utils.box_formats import yolo_xywh_to_xyxy_tensor


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


def _yolo_box_valid(box: list[float]) -> bool:
    if len(box) != 4:
        return False
    cx, cy, w, h = [float(v) for v in box]
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


class PromptEpisodeDataset(Dataset):
    def __init__(
        self,
        image_list_path: str,
        labels_dir: str,
        class_names_path: str,
        image_size: int,
        episodes_per_epoch: int,
        negative_ratio: float = 0.2,
        hard_negative_ratio: float = 0.2,
        prompt_type: str = "mixed",
        min_prompt_classes: int = 1,
        max_prompt_classes: int = 3,
        max_prompt_instances_per_class: int = 2,
        max_prompt_images: int = 4,
        confusable_non_target_weight: float = 2.0,
        color_min_distance: float = 0.45,
        same_instance_ratio: float = 0.1,
        center_target_sigma: float = 0.35,
        hint_inner_shrink: float = 0.6,
        hint_bg_expand: float = 0.12,
        seed: int | None = None,
    ):
        super().__init__()
        self.labels_dir = Path(labels_dir)
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
        self.color_min_distance = color_min_distance
        self.same_instance_ratio = same_instance_ratio
        self.center_target_sigma = center_target_sigma
        self.hint_inner_shrink = hint_inner_shrink
        self.hint_bg_expand = hint_bg_expand
        self.seed = seed

        self.image_records: Dict[int, dict] = {}
        self.image_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.image_to_class_ids: Dict[int, set[int]] = defaultdict(set)
        self.class_to_anns: Dict[int, List[dict]] = defaultdict(list)
        self.class_to_image_ids: Dict[int, set[int]] = defaultdict(set)
        self.class_image_to_anns: Dict[int, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
        self.categories = load_class_names(class_names_path)
        self.class_to_family: Dict[int, str] = {}
        self.family_to_classes: Dict[str, set[int]] = defaultdict(set)
        self.family_confusions: Dict[str, set[str]] = {
            "diamond": {"triangle"},
            "triangle": {"diamond"},
        }
        for category_id, category_name in self.categories.items():
            family = category_name.split("_")[-1] if "_" in category_name else category_name
            self.class_to_family[category_id] = family
            self.family_to_classes[family].add(category_id)
        ann_id = 0
        for image_id, image_path in enumerate(load_image_list(image_list_path)):
            with Image.open(image_path) as image:
                image_w, image_h = image.size
            self.image_records[image_id] = {
                "id": image_id,
                "file_name": image_path.name,
                "path": str(image_path),
                "width": image_w,
                "height": image_h,
            }
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            for category_id, bbox in parse_yolo_label_file(label_path):
                if category_id not in self.categories or not _yolo_box_valid(bbox):
                    continue
                bbox_xyxy = yolo_xywh_to_xyxy_tensor(
                    torch.tensor([bbox], dtype=torch.float32),
                    image_w,
                    image_h,
                )[0].tolist()
                ann_record = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox_xyxy,
                }
                ann_id += 1
                self.image_to_anns[image_id].append(ann_record)
                self.image_to_class_ids[image_id].add(category_id)
                self.class_to_anns[category_id].append(ann_record)
                self.class_to_image_ids[category_id].add(image_id)
                self.class_image_to_anns[category_id][image_id].append(ann_record)

        self.image_ids = sorted(self.image_records.keys())
        self.classes = sorted(self.class_to_anns.keys())

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _load_image(self, image_id: int) -> Image.Image:
        path = Path(self.image_records[image_id]["path"])
        return Image.open(path).convert("RGB")

    def _choose_prompt_type(self, rng: random.Random) -> int:
        if self.prompt_type == "same_category":
            return 0
        if self.prompt_type == "same_instance":
            return 1
        return 1 if rng.random() < self.same_instance_ratio else 0

    def _sample_prompt_classes(self, prompt_type_id: int, rng: random.Random) -> List[int]:
        if prompt_type_id == 1:
            return [rng.choice(self.classes)]
        max_classes = min(self.max_prompt_classes, len(self.classes))
        min_classes = min(self.min_prompt_classes, max_classes)
        count = rng.randint(min_classes, max_classes)
        return rng.sample(self.classes, count)

    def _sample_prompt_instances(
        self,
        prompt_class_ids: List[int],
        class_to_slot: Dict[int, int],
        rng: random.Random,
    ) -> List[dict]:
        selected_instances: List[dict] = []
        used_image_ids: set[int] = set()
        used_ann_ids: set[int] = set()
        for class_id in prompt_class_ids:
            class_slot = class_to_slot[class_id]
            image_ids = list(self.class_to_image_ids[class_id])
            rng.shuffle(image_ids)
            max_instances = min(self.max_prompt_instances_per_class, max(len(self.class_to_anns[class_id]), 1))
            target_instances = rng.randint(1, max_instances)
            class_instances: List[dict] = []
            for image_id in image_ids:
                if len(class_instances) >= target_instances:
                    break
                if len(used_image_ids) >= self.max_prompt_images and image_id not in used_image_ids:
                    continue
                candidates = [ann for ann in self.class_image_to_anns[class_id][image_id] if ann["id"] not in used_ann_ids]
                if not candidates:
                    candidates = self.class_image_to_anns[class_id][image_id]
                ann = rng.choice(candidates)
                used_image_ids.add(image_id)
                used_ann_ids.add(ann["id"])
                class_instances.append({
                    "image_id": image_id,
                    "bbox": ann["bbox"],
                    "class_slot": class_slot,
                    "category_id": class_id,
                })
            if not class_instances:
                ann = rng.choice(self.class_to_anns[class_id])
                class_instances.append({
                    "image_id": ann["image_id"],
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
            for related_family in self.family_confusions.get(family, set()):
                confusable.update(self.family_to_classes.get(related_family, set()) - prompt_class_set)
        return confusable

    def _score_positive_query(self, image_id: int, prompt_class_set: set[int]) -> tuple[int, int, int, int]:
        anns = self.image_to_anns[image_id]
        cover = len(self.image_to_class_ids[image_id] & prompt_class_set)
        target_instances = sum(1 for ann in anns if ann["category_id"] in prompt_class_set)
        unrelated = sum(1 for ann in anns if ann["category_id"] not in prompt_class_set)
        confusable = sum(1 for ann in anns if ann["category_id"] in self._get_confusable_classes(list(prompt_class_set)))
        return cover, target_instances, -unrelated, -confusable

    def _select_positive_query(
        self,
        prompt_class_ids: List[int],
        prompt_image_ids: List[int],
        allow_same_image: bool,
        rng: random.Random,
    ) -> int | None:
        prompt_class_set = set(prompt_class_ids)
        prompt_image_set = set(prompt_image_ids)
        candidates = sorted(set().union(*(self.class_to_image_ids[class_id] for class_id in prompt_class_ids)))
        if not allow_same_image:
            candidates = [image_id for image_id in candidates if image_id not in prompt_image_set]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda image_id: self._score_positive_query(image_id, prompt_class_set), reverse=True)
        shortlist = ranked[: min(len(ranked), 3)]
        return rng.choice(shortlist)

    def _select_negative_query(
        self,
        prompt_class_ids: List[int],
        prompt_image_ids: List[int],
        rng: random.Random,
    ) -> int | None:
        prompt_class_set = set(prompt_class_ids)
        prompt_image_set = set(prompt_image_ids)
        confusable_class_set = self._get_confusable_classes(prompt_class_ids)
        negative_ids = [
            image_id
            for image_id in self.image_ids
            if not (self.image_to_class_ids[image_id] & prompt_class_set) and image_id not in prompt_image_set
        ]
        if not negative_ids:
            return None
        confusable_negative_ids = [
            image_id
            for image_id in negative_ids
            if self.image_to_class_ids[image_id] & confusable_class_set
        ]
        hard_negative_ids = [image_id for image_id in negative_ids if len(self.image_to_class_ids[image_id]) > 0]
        sample = rng.random()
        if sample < self.hard_negative_ratio:
            pool = confusable_negative_ids or hard_negative_ids or negative_ids
        elif sample < self.hard_negative_ratio + self.negative_ratio:
            pool = hard_negative_ids or negative_ids
        else:
            return None
        pool = sorted(pool, key=lambda image_id: len(self.image_to_anns[image_id]), reverse=True)
        return rng.choice(pool[: min(len(pool), 3)])

    def _sample_query(
        self,
        prompt_type_id: int,
        prompt_class_ids: List[int],
        prompt_image_ids: List[int],
        rng: random.Random,
    ) -> Tuple[int, bool]:
        positive_id = self._select_positive_query(prompt_class_ids, prompt_image_ids, allow_same_image=False, rng=rng)
        negative_id = self._select_negative_query(prompt_class_ids, prompt_image_ids, rng=rng)

        if prompt_type_id == 1:
            if positive_id is not None:
                return positive_id, True
            fallback_positive = self._select_positive_query(prompt_class_ids, prompt_image_ids, allow_same_image=True, rng=rng)
            if fallback_positive is not None:
                return fallback_positive, True

        if negative_id is not None:
            return negative_id, False
        if positive_id is not None:
            return positive_id, True

        fallback_positive = self._select_positive_query(prompt_class_ids, prompt_image_ids, allow_same_image=True, rng=rng)
        if fallback_positive is not None:
            return fallback_positive, True
        return rng.choice(self.image_ids), False

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index) if self.seed is not None else random
        prompt_type_id = self._choose_prompt_type(rng)
        sampled_prompt_class_ids = self._sample_prompt_classes(prompt_type_id, rng)
        slot_order = rng.sample(range(len(sampled_prompt_class_ids)), len(sampled_prompt_class_ids))
        class_to_slot = {class_id: slot_order[idx] for idx, class_id in enumerate(sampled_prompt_class_ids)}
        prompt_class_ids = [-1] * len(sampled_prompt_class_ids)
        for class_id, slot in class_to_slot.items():
            prompt_class_ids[slot] = class_id

        prompt_instances = self._sample_prompt_instances(sampled_prompt_class_ids, class_to_slot, rng)
        prompt_image_ids = [instance["image_id"] for instance in prompt_instances]
        query_image_id, positive = self._sample_query(prompt_type_id, sampled_prompt_class_ids, prompt_image_ids, rng)

        color_generator = None
        if self.seed is not None:
            color_generator = torch.Generator().manual_seed(self.seed + index)
        slot_colors = sample_slot_colors(
            len(sampled_prompt_class_ids),
            min_distance=self.color_min_distance,
            generator=color_generator,
        )
        prompt_images = []
        prompt_boxes = []
        prompt_hint_maps = []
        prompt_target_maps = []
        prompt_class_indices = []
        for instance in prompt_instances:
            image = self._load_image(instance["image_id"])
            box = torch.tensor([instance["bbox"]], dtype=torch.float32)
            image_tensor, box = resize_image_and_boxes(image, box, self.image_size)
            hint = build_prompt_hint_map(
                self.image_size,
                box[0],
                inner_shrink=self.hint_inner_shrink,
                bg_expand=self.hint_bg_expand,
            )
            prompt_images.append(image_tensor)
            prompt_boxes.append(box[0])
            prompt_hint_maps.append(hint)
            prompt_target_maps.append(
                build_prompt_target_map(
                    self.image_size,
                    box[0],
                    int(instance["class_slot"]),
                    slot_colors,
                    center_sigma=self.center_target_sigma,
                )
            )
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
        query_dense_slot_target, query_dense_fg_target, query_dense_center_target, query_dense_valid_mask, query_target_map = build_query_detection_targets(
            self.image_size,
            query_boxes_tensor,
            query_labels_tensor,
            slot_colors,
            center_sigma=self.center_target_sigma,
        )

        return {
            "prompt_images": torch.stack(prompt_images, dim=0),
            "prompt_boxes": torch.stack(prompt_boxes, dim=0),
            "prompt_hint_maps": torch.stack(prompt_hint_maps, dim=0),
            "prompt_target_maps": torch.stack(prompt_target_maps, dim=0),
            "prompt_class_indices": torch.tensor(prompt_class_indices, dtype=torch.long),
            "prompt_class_ids": torch.tensor(prompt_class_ids, dtype=torch.long),
            "slot_colors": slot_colors,
            "prompt_type": torch.tensor(prompt_type_id, dtype=torch.long),
            "query_image": query_image_tensor,
            "query_boxes": query_boxes_tensor,
            "query_labels": query_labels_tensor,
            "query_category_ids": query_category_ids_tensor,
            "query_dense_slot_target": query_dense_slot_target,
            "query_dense_fg_target": query_dense_fg_target,
            "query_dense_center_target": query_dense_center_target,
            "query_dense_valid_mask": query_dense_valid_mask,
            "query_target_map": query_target_map,
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
    target_channels = batch[0]["prompt_target_maps"].shape[1]

    prompt_images = torch.zeros((batch_size, max_prompt_instances, channels, height, width), dtype=torch.float32)
    prompt_boxes = torch.zeros((batch_size, max_prompt_instances, 4), dtype=torch.float32)
    prompt_hint_maps = torch.zeros((batch_size, max_prompt_instances, 3, height, width), dtype=torch.float32)
    prompt_target_maps = torch.zeros((batch_size, max_prompt_instances, target_channels, height, width), dtype=torch.float32)
    prompt_class_indices = torch.zeros((batch_size, max_prompt_instances), dtype=torch.long)
    prompt_instance_mask = torch.zeros((batch_size, max_prompt_instances), dtype=torch.bool)
    prompt_class_ids = torch.full((batch_size, max_prompt_classes), -1, dtype=torch.long)
    prompt_class_mask = torch.zeros((batch_size, max_prompt_classes), dtype=torch.bool)
    slot_colors = torch.zeros((batch_size, max_prompt_classes, 3), dtype=torch.float32)

    for batch_idx, item in enumerate(batch):
        num_instances = item["prompt_images"].shape[0]
        num_classes = item["prompt_class_ids"].shape[0]
        prompt_images[batch_idx, :num_instances] = item["prompt_images"]
        prompt_boxes[batch_idx, :num_instances] = item["prompt_boxes"]
        prompt_hint_maps[batch_idx, :num_instances] = item["prompt_hint_maps"]
        prompt_target_maps[batch_idx, :num_instances] = item["prompt_target_maps"]
        prompt_class_indices[batch_idx, :num_instances] = item["prompt_class_indices"]
        prompt_instance_mask[batch_idx, :num_instances] = True
        prompt_class_ids[batch_idx, :num_classes] = item["prompt_class_ids"]
        prompt_class_mask[batch_idx, :num_classes] = True
        slot_colors[batch_idx, :num_classes] = item["slot_colors"]

    return {
        "prompt_images": prompt_images,
        "prompt_boxes": prompt_boxes,
        "prompt_hint_maps": prompt_hint_maps,
        "prompt_target_maps": prompt_target_maps,
        "prompt_class_indices": prompt_class_indices,
        "prompt_instance_mask": prompt_instance_mask,
        "prompt_class_ids": prompt_class_ids,
        "prompt_class_mask": prompt_class_mask,
        "slot_colors": slot_colors,
        "prompt_type": torch.stack([item["prompt_type"] for item in batch], dim=0),
        "query_image": torch.stack([item["query_image"] for item in batch], dim=0),
        "query_target_map": torch.stack([item["query_target_map"] for item in batch], dim=0),
        "targets": [
            {
                "boxes": item["query_boxes"],
                "labels": item["query_labels"],
                "category_ids": item["query_category_ids"],
                "dense_slot_target": item["query_dense_slot_target"],
                "dense_fg_target": item["query_dense_fg_target"],
                "dense_center_target": item["query_dense_center_target"],
                "dense_valid_mask": item["query_dense_valid_mask"],
                "query_target_map": item["query_target_map"],
                "non_target_boxes": item["query_non_target_boxes"],
                "non_target_weights": item["query_non_target_weights"],
                "image_size": int(item["image_size"].item()),
            }
            for item in batch
        ],
        "is_positive": torch.stack([item["is_positive"] for item in batch], dim=0),
    }
