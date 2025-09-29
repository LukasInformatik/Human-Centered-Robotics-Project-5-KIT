# person_tracker.py
import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class HumanTracker:
    """
    YOLOv8(+BoT-SORT) person tracker with mouse click selection.
    Secondary ReID uses YOLO backbone features.
    - .track(rgb) -> (bbox, track_id), bbox = [x, y, w, h]
    """

    # ---- Constants ----
    EMBED_INPUT_SIZE: int = 224     # crop size for feature extraction
    MISS_PATIENCE: int = 10         # frames without target before ReID
    SIM_THRESHOLD: float = 0.45     # cosine threshold for ReID
    EMA_MOMENTUM: float = 0.9       # smoothing factor for target embedding

    # ---- UI ----
    WINDOW_NAME: str = "Tracked"

    def __init__(self) -> None:
        # get model path from ROS package
        package_share = get_package_share_directory("project_5_pkg")
        model_path = os.path.join(package_share, "models", "yolov8n_80_epochs.pt")

        # load YOLOv8 detector
        self.model = YOLO(model_path)
        self.model.model.eval()

        # state: selection & last detections
        self.selected_id: Optional[int] = None
        self.latest_boxes: List = []

        # display + mouse callback
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse_click)

        # ReID setup
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)

        # hooked feature map
        self._feature = None

        # smoothed target embedding
        self._target_embed_avg: Optional[np.ndarray] = None

        # miss counter for recovery
        self._miss_streak: int = 0

        # hook a late backbone layer
        self._register_feature_hook()

    # ---------------- UI / Selection ----------------
    def _on_mouse_click(self, event: int, x: int, y: int, flags, param=None) -> None:
        """Left click on a box -> set selected ID and reset ReID state."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for box in self.latest_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                self.selected_id = int(box.id[0])
                self._target_embed_avg = None
                self._miss_streak = 0
                print(f"[INFO] Selected track ID: {self.selected_id}")
                break

    # ---------------- ReID: Feature Hook & Embedding ----------------
    def _register_feature_hook(self) -> None:
        """Register a forward hook on a late backbone layer."""
        def _hook(_module, _inp, output):
            self._feature = output

        try:
            self.model.model.model[-2].register_forward_hook(_hook)
        except Exception:
            self.model.model.model[-1].register_forward_hook(_hook)

    def _compute_embedding(self, rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Crop box, forward pass (hook captures feature),
        global average pool, L2-normalize.
        """
        h, w = rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.EMBED_INPUT_SIZE, self.EMBED_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop).float().to(self._device) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.inference_mode():
            _ = self.model.model(tensor)

        feature = self._feature
        if feature is None:
            return None
        if isinstance(feature, (list, tuple)):
            feature = feature[0]

        vec = F.adaptive_avg_pool2d(feature, 1).squeeze().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def _ema_update_target(self, new_embed: np.ndarray) -> None:
        """Update target embedding with EMA and re-normalize."""
        if new_embed is None:
            return
        if self._target_embed_avg is None:
            self._target_embed_avg = new_embed
            return
        avg = self.EMA_MOMENTUM * self._target_embed_avg + (1.0 - self.EMA_MOMENTUM) * new_embed
        self._target_embed_avg = avg / (np.linalg.norm(avg) + 1e-12)

    # ---------------- Tracking ----------------
    def track(self, rgb: np.ndarray) -> Tuple[Optional[List[int]], Optional[int]]:
        """
        Run detection + BoT-SORT tracking,
        draw boxes, run ReID recovery if needed.
        """
        results = self.model.track(rgb, conf=0.25, tracker="botsort.yaml", persist=True)

        self.latest_boxes = []
        out_bbox: Optional[List[int]] = None
        out_id: Optional[int] = None
        target_visible = False

        # process detections
        for r in results:
            self.latest_boxes = r.boxes
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1

                # if selected ID visible: update bbox + embedding
                if self.selected_id is not None and tid == self.selected_id:
                    out_bbox = [x1, y1, x2 - x1, y2 - y1]
                    out_id = tid
                    target_visible = True

                    embed = self._compute_embedding(rgb, x1, y1, x2, y2)
                    if embed is not None:
                        self._ema_update_target(embed)

                # visualization
                color = (0, 0, 255) if tid == self.selected_id else (0, 255, 0)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ReID recovery if target missing
        if self.selected_id is not None and self._target_embed_avg is not None:
            if target_visible:
                self._miss_streak = 0
            else:
                self._miss_streak += 1
                if self._miss_streak >= self.MISS_PATIENCE and self.latest_boxes:
                    best_sim = -1.0
                    best_tid: Optional[int] = None
                    best_bbox: Optional[List[int]] = None

                    for box in self.latest_boxes:
                        if box.id is None:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cand_embed = self._compute_embedding(rgb, x1, y1, x2, y2)
                        if cand_embed is None:
                            continue
                        sim = float(np.dot(self._target_embed_avg, cand_embed))
                        if sim > best_sim:
                            best_sim = sim
                            best_tid = int(box.id[0])
                            best_bbox = [x1, y1, x2 - x1, y2 - y1]

                    if best_tid is not None and best_sim >= self.SIM_THRESHOLD:
                        self.selected_id = best_tid
                        out_bbox, out_id = best_bbox, best_tid
                        self._miss_streak = 0
                        print(f"[INFO] ReID recovery: selected_id -> {best_tid} (sim={best_sim:.2f})")

        # display
        cv2.imshow(self.WINDOW_NAME, rgb)
        cv2.waitKey(1)

        return out_bbox, out_id
