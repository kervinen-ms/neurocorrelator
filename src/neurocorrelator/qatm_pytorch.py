from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from seaborn import color_palette
from torchvision import transforms

from neurocorrelator.featex import VGGFeatureExtractor

from .utils import *


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for template matching with QATM"""

    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        template_dir: Path,
        image_path: Path,
        thresh_csv: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            template_dir (Path): template images directory
            image_path (Path): sample image
            thresh_csv (Optional[Path], optional): confidence thresholds for templates. Defaults to None.
            transform (Optional[Callable], optional): transformation to apply to images. Defaults to None.
        """

        self.transform = transform or self.DEFAULT_TRANSFORM
        self.template_paths = list(template_dir.glob("*"))
        self.image_path = image_path
        self.image_raw = cv2.imread(image_path)
        self.thresholds = self._load_thresholds(thresh_csv) if thresh_csv else {}
        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)

    def _load_thresholds(self, csv_path: Path) -> Dict[Path, float]:
        """Load template thresholds from CSV

        Args:
            csv_path (Path): path to thresholds csv

        Returns:
            Dict[Path, float]: template path -> threshold mapping
        """

        if not csv_path.exists():
            return {}
        df = pd.read_csv(csv_path)
        return {Path(row["path"]): row["thresh"] for _, row in df.iterrows()}

    def __len__(self) -> int:
        return len(self.template_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        template_path = self.template_paths[idx]
        template_img = cv2.imread(str(template_path))

        if self.transform:
            template = self.transform(template_img)

        threshold = self.thresholds.get(template_path, 0.7)

        return {
            "image": self.image,
            "image_raw": self.image_raw,
            "image_path": self.image_path,
            "template": template.unsqueeze(0),
            "template_path": template_path,
            "template_h": template.shape[-2],
            "template_w": template.shape[-1],
            "threshold": threshold,
        }


# class FeatureExtractor(nn.Module):
#     """Feature extractor for VGG19 with hook-based feature capture"""

#     def __init__(self, base_model: nn.Module, use_cuda: bool = False):
#         super().__init__()
#         self.use_cuda = use_cuda
#         self.model = self._create_feature_extractor(base_model)
#         self.feature1: torch.Tensor = None
#         self.feature2: torch.Tensor = None

#         if use_cuda:
#             self.model.cuda()

#     def _create_feature_extractor(self, model: nn.Module) -> nn.Module:
#         """Truncate model and register hooks"""
#         model = model[:17]
#         for param in model.parameters():
#             param.requires_grad = False

#         model[2].register_forward_hook(self._save_feature1)
#         model[16].register_forward_hook(self._save_feature2)
#         return model.eval()

#     def _save_feature1(self, module, input, output):
#         self.feature1 = output.detach()

#     def _save_feature2(self, module, input, output):
#         self.feature2 = output.detach()

#     def forward(self, x: torch.Tensor, mode: str = "big") -> torch.Tensor:
#         if self.use_cuda:
#             x = x.cuda()

#         self.model(x)

#         if mode == "big":
#             target_size = (self.feature2.shape[2], self.feature2.shape[3])
#             feature = F.interpolate(
#                 self.feature1, size=target_size, mode="bilinear", align_corners=True
#             )
#         else:
#             target_size = (self.feature1.shape[2], self.feature1.shape[3])
#             feature = F.interpolate(
#                 self.feature2, size=target_size, mode="bilinear", align_corners=True
#             )

#         return torch.cat(
#             (feature, self.feature2 if mode == "big" else self.feature1), dim=1
#         )


class FeatureNormalizer:
    """Normalizes features across both input tensors"""

    def __call__(self, feature1: torch.Tensor, feature2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, h1, w1 = feature1.size()
        _, _, h2, w2 = feature2.size()
        feature1 = feature1.view(bs, -1, h1 * w1)
        feature2 = feature2.view(bs, -1, h2 * w2)
        combined = torch.cat([feature1, feature2], dim=2)
        mean = combined.mean(dim=2, keepdim=True)
        std = combined.std(dim=2, keepdim=True)

        feature1 = (feature1 - mean) / (std + 1e-12)
        feature2 = (feature2 - mean) / (std + 1e-12)
        feature1 = feature1.view(bs, -1, h1, w1)
        feature2 = feature2.view(bs, -1, h2, w2)
        return feature1, feature2


class QATMModel(nn.Module):
    """QATM Model for template matching"""

    def __init__(self, alpha: float, use_trained=False, use_cuda: bool = False):
        super().__init__()
        self.alpha = alpha
        self.featex = VGGFeatureExtractor()
        if use_trained:
            self.featex.load_state_dict(torch.load("model"))
        self.normalize = FeatureNormalizer()
        self.qatm = QATM(alpha)
        self.cached_features: Dict[Path, torch.Tensor] = {}

    def forward(self, template: torch.Tensor, image: torch.Tensor, image_path: Path) -> torch.Tensor:
        T_feat = self.featex(template)

        if image_path not in self.cached_features:
            self.cached_features[image_path] = self.featex(image)
        I_feat = self.cached_features[image_path]

        conf_maps = []
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_norm, T_norm = self.normalize(I_feat, T_feat_i)

            # Compute cosine similarity
            I_norm = I_norm / (torch.norm(I_norm, dim=1, keepdim=True) + 1e-12)
            T_norm = T_norm / (torch.norm(T_norm, dim=1, keepdim=True) + 1e-12)
            dist = torch.einsum("xcab,xcde->xabde", I_norm, T_norm)

            conf_maps.append(self.qatm(dist))

        return torch.cat(conf_maps, dim=0)


class QATM(nn.Module):
    """Quality-Aware Template Matching layer"""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): similarity-score tensor

        Returns:
            torch.Tensor: qatm-score tensor
        """
        b, h, w, h2, w2 = x.shape
        x_flat = x.view(b, h * w, h2 * w2)

        # Compute softmax along reference and query dimensions
        x_ref = x_flat - x_flat.max(dim=1, keepdim=True)[0]
        x_qry = x_flat - x_flat.max(dim=2, keepdim=True)[0]

        soft_ref = F.softmax(self.alpha * x_ref, dim=1)
        soft_qry = F.softmax(self.alpha * x_qry, dim=2)

        confidence = torch.sqrt(soft_ref * soft_qry)
        return confidence.max(dim=2, keepdim=True)[0].view(b, h, w, 1)


def nms_single(score: np.ndarray, template_w: int, template_h: int, threshold: float = 0.7) -> np.ndarray:
    """Non-Maximum Suppression for single template"""
    y, x = np.where(score > threshold * score.max())
    boxes = np.array(
        [
            x - template_w // 2,
            y - template_h // 2,
            x + template_w // 2,
            y + template_h // 2,
        ]
    ).T

    if boxes.size == 0:
        return np.empty((0, 4))

    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    scores = score[y, x]
    indices = scores.argsort()[::-1]

    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = w * h
        iou = overlap / (areas[i] + areas[indices[1:]] - overlap)

        indices = indices[iou <= 0.5][1:]

    return boxes[keep]


def nms_multi(
    scores: np.ndarray,
    template_widths: np.ndarray,
    template_heights: np.ndarray,
    thresholds: list[float],
    iou_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-Maximum Suppression for multiple templates with confidence scores

    Args:
        scores: Confidence scores array of shape (num_templates, H, W)
        template_widths: Array of template widths
        template_heights: Array of template heights
        thresholds: List of threshold values for each template
        iou_threshold: IoU threshold for NMS (default: 0.5)

    Returns:
        boxes: Detected bounding boxes in [[x1, y1], [x2, y2]] format per box
        indices: Template indices for each detected box
        confidences: Confidence scores for each detected box
    """
    max_scores = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    valid_mask = max_scores > 0.1 * max_scores.max()
    valid_indices = np.where(valid_mask)[0]

    if not valid_indices.size:
        return np.empty((0, 2, 2)), np.array([], dtype=int), np.array([], dtype=float)

    candidate_points = []
    candidate_indices = []

    for idx in valid_indices:
        score_map = scores[idx]
        threshold = thresholds[idx]
        y, x = np.where(score_map > threshold * score_map.max())

        if y.size:
            for point in zip(y, x):
                candidate_points.append(point)
                candidate_indices.append(idx)

    if not candidate_points:
        return np.empty((0, 2, 2)), np.array([], dtype=int), np.array([], dtype=float)

    y, x = np.array(candidate_points).T
    candidate_indices = np.array(candidate_indices)
    widths = template_widths[candidate_indices]
    heights = template_heights[candidate_indices]

    boxes = np.empty((len(x), 4))
    boxes[:, 0] = x - widths // 2
    boxes[:, 1] = y - heights // 2
    boxes[:, 2] = x + widths // 2
    boxes[:, 3] = y + heights // 2

    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    point_scores = scores[candidate_indices, y, x]

    indices = point_scores.argsort()[::-1]
    keep = []

    while indices.size > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = w * h
        iou = overlap / (areas[i] + areas[indices[1:]] - overlap + 1e-10)

        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return boxes[keep].reshape(-1, 2, 2), candidate_indices[keep], point_scores[keep]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    template_indices: Optional[np.ndarray] = None,
    show: bool = False,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Draws predicted boxes on the sample image

    Args:
        image (np.ndarray): sample image
        boxes (np.ndarray): predictions bounding boxes
        confidence_scores (Optional[np.ndarray], optional): QATM scores of boxes. Defaults to None.
        template_indices (Optional[np.ndarray], optional): templates which boxes are predicted. Defaults to None.
        show (bool, optional): whether to show image or not. Defaults to False.
        save_path (Optional[Path], optional): path to image with boxes drawn. Defaults to None.

    Returns:
        np.ndarray: images with boxes
    """
    img_out = image.copy()
    color_list = None

    if template_indices is not None:
        colors = color_palette("hls", template_indices.max() + 1)
        color_list = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    for i, box in enumerate(boxes):
        color = color_list[template_indices[i]] if color_list else (0, 0, 255)
        (x1, y1), (x2, y2) = np.astype(box, int)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        if confidence_scores is not None:
            label = f"{confidence_scores[i]:.2f}"
            cv2.putText(
                img_out,
                label,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,
                (255, 255, 255),
                1,
            )

    if show:
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if save_path:
        cv2.imwrite(str(save_path), img_out)

    return img_out


def process_single_template(
    model: QATMModel,
    template: torch.Tensor,
    image: torch.Tensor,
    image_path: Path,
    template_size: Tuple[int, int],
) -> np.ndarray:
    """Process single template and return score map"""
    with torch.no_grad():
        val = model(template, image, image_path)
        val = val.cpu().numpy() if val.is_cuda else val.numpy()

    log_score = np.log(val)

    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        gray = cv2.resize(log_score[i, :, :, 0], (image.shape[-1], image.shape[-2]))
        h, w = template_size
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))
        scores.append(score)
    scores = np.array(scores)
    return scores, scores.sum(axis=0)


def process_dataset(model: QATMModel, dataset: ImageDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Process entire dataset and return results"""
    scores = []
    score_maps = []
    template_sizes = []
    thresholds = []

    for data in dataset:
        score, score_map = process_single_template(
            model,
            data["template"],
            data["image"],
            data["image_path"],
            (data["template_h"], data["template_w"]),
        )
        scores.append(score)
        score_maps.append(score_map)
        template_sizes.append((data["template_w"], data["template_h"]))
        thresholds.append(data["threshold"])

    widths, heights = zip(*template_sizes)
    return (
        np.squeeze(np.array(scores), axis=1),
        np.array(widths),
        np.array(heights),
        thresholds,
        np.array(score_maps),
    )


def run_pipeline(
    template_dir: Path,
    image_path: Path,
    result_dir: Path,
    alpha: float = 25,
    use_cuda: bool = True,
    threshold_csv: Optional[Path] = None,
    use_trained=False,
):
    """Full QATM pipeline from input to result visualization"""

    # Prepare dataset
    dataset = ImageDataset(template_dir, image_path, threshold_csv)

    # Create and run model
    model = QATMModel(alpha, use_cuda=use_cuda, use_trained=use_trained)

    scores, widths, heights, thresholds, heatmap = process_dataset(model, dataset)
    boxes, indices, confidences = nms_multi(scores, widths, heights, thresholds)

    # Visualize results
    result_img = draw_boxes(
        dataset.image_raw,
        boxes,
        confidences,
        indices,
        save_path=result_dir / "result.png",
    )

    # Save heatmap
    heatmap = heatmap.sum(axis=0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(result_dir / "heatmap.png", heatmap)

    # Save bounding boxes
    boxes_df = pd.DataFrame(boxes.reshape(-1, 4), columns=["x1", "y1", "x2", "y2"], dtype=int)
    boxes_df.to_csv(result_dir / "boxes.csv", sep="\t")
    return boxes, indices, confidences, result_img
