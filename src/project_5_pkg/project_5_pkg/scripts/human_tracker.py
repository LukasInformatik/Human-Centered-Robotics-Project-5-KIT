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
    YOLOv8(+BoT-SORT) Personentracker mit Mausklick-Auswahl.
    Sekundäre ReID-Schicht nutzt Backbone-Features desselben YOLO-Modells.
    - .track(rgb) -> (bbox, track_id), wobei bbox = [x, y, w, h]
    """

    # ---- Tuning-Konstanten ----
    EMBED_INPUT_SIZE: int = 224     # Größe der Crops für die Feature-Extraktion
    MISS_PATIENCE: int = 10         # #Frames ohne Sichtkontakt, bevor ReID-Recovery startet
    SIM_THRESHOLD: float = 0.45     # Cosine-Schwelle für ReID-Zuordnung (0.35–0.60 anpassen)
    EMA_MOMENTUM: float = 0.9       # Glättung (Exponential Moving Average) für das Ziel-Embedding

    # ---- UI ----
    WINDOW_NAME: str = "Tracked"

    def __init__(self) -> None:
        # Modellpfad aus ROS-Package holen
        package_share = get_package_share_directory("project_5_pkg")
        model_path = os.path.join(package_share, "models", "yolov8n_80_epochs.pt")

        # YOLOv8-Detektor laden (Ultralytics managed intern PyTorch)
        self.model = YOLO(model_path)
        self.model.model.eval()  # sicherheitshalber in Eval-Modus

        # State: Auswahl & letzte Detections
        self.selected_id: Optional[int] = None
        self.latest_boxes: List = []

        # Anzeige + Mausauswahl
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse_click)

        # -------- ReID-Setup (Backbone-Feature-Hook) --------
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)

        # Hier landet die zuletzt „gehookte“ Feature-Map des Backbones
        self._feature = None

        # Gleitender Mittelwert (L2-normalisiert) des Ziel-Embeddings
        self._target_embed_avg: Optional[np.ndarray] = None

        # Miss-Counter für Recovery
        self._miss_streak: int = 0

        # Hook an geeignete Schicht registrieren (vorletzte, Fallback letzte)
        self._register_feature_hook()

    # --------------------------------------------------------------------- #
    #                            UI / Auswahl                               #
    # --------------------------------------------------------------------- #
    def _on_mouse_click(self, event: int, x: int, y: int, flags, param=None) -> None:
        """Linksklick auf eine Box -> setzt die ausgewählte Track-ID und resettet ReID-Status."""
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

    # --------------------------------------------------------------------- #
    #                      ReID: Feature-Hook & Embedding                   #
    # --------------------------------------------------------------------- #
    def _register_feature_hook(self) -> None:
        """Registriert einen Forward-Hook auf eine späte Backbone-Schicht."""
        def _hook(_module, _inp, output):
            self._feature = output

        # YOLOv8-Modelle: häufig ist [-2] passend; als Fallback [-1]
        try:
            self.model.model.model[-2].register_forward_hook(_hook)
        except Exception:
            self.model.model.model[-1].register_forward_hook(_hook)

    def _compute_embedding(self, rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Schneidet die Box, führt sie einmal durch das Modell (Backbone-Feature via Hook),
        pooled global zu einem Vektor und L2-normalisiert diesen.
        """
        h, w = rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize -> Tensor [1,3,H,W] in [0,1]
        crop = cv2.resize(crop, (self.EMBED_INPUT_SIZE, self.EMBED_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop).float().to(self._device) / 255.0         # HWC
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)                            # NCHW

        # Ein kurzer Forward (kein NMS/Detektion nötig) -> Hook setzt self._feature
        with torch.inference_mode():
            _ = self.model.model(tensor)

        feature = self._feature
        if feature is None:
            return None
        if isinstance(feature, (list, tuple)):   # falls Modell mehrere Maps liefert
            feature = feature[0]

        # Global Average Pooling -> [C] und L2-Norm
        vec = F.adaptive_avg_pool2d(feature, 1).squeeze().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def _ema_update_target(self, new_embed: np.ndarray) -> None:
        """Aktualisiert das Ziel-Embedding per Exponential Moving Average + Re-Norm."""
        if new_embed is None:
            return
        if self._target_embed_avg is None:
            self._target_embed_avg = new_embed
            return
        avg = self.EMA_MOMENTUM * self._target_embed_avg + (1.0 - self.EMA_MOMENTUM) * new_embed
        self._target_embed_avg = avg / (np.linalg.norm(avg) + 1e-12)

    # --------------------------------------------------------------------- #
    #                              Tracking                                 #
    # --------------------------------------------------------------------- #
    def track(self, rgb: np.ndarray) -> Tuple[Optional[List[int]], Optional[int]]:
        """
        Führt Detection + Tracking (BoT-SORT via botsort.yaml) durch,
        zeichnet Boxen und macht ggf. ReID-Recovery.
        """
        results = self.model.track(rgb, conf=0.25, tracker="botsort.yaml", persist=True)

        self.latest_boxes = []
        out_bbox: Optional[List[int]] = None
        out_id: Optional[int] = None
        target_visible = False

        # ---- Detections/Tracks verarbeiten ----
        for r in results:
            self.latest_boxes = r.boxes
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1

                # Falls die ausgewählte ID sichtbar ist: bbox ausgeben + ReID-Embedding sammeln
                if self.selected_id is not None and tid == self.selected_id:
                    out_bbox = [x1, y1, x2 - x1, y2 - y1]
                    out_id = tid
                    target_visible = True

                    embed = self._compute_embedding(rgb, x1, y1, x2, y2)
                    if embed is not None:
                        self._ema_update_target(embed)

                # Visualisierung
                color = (0, 0, 255) if tid == self.selected_id else (0, 255, 0)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- ReID-Recovery: Ziel fehlt seit MISS_PATIENCE Frames ----
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
                        sim = float(np.dot(self._target_embed_avg, cand_embed))  # Cosine bei L2-normierten Vektoren
                        if sim > best_sim:
                            best_sim = sim
                            best_tid = int(box.id[0])
                            best_bbox = [x1, y1, x2 - x1, y2 - y1]  # direkt korrekt setzen

                    if best_tid is not None and best_sim >= self.SIM_THRESHOLD:
                        self.selected_id = best_tid
                        out_bbox, out_id = best_bbox, best_tid
                        self._miss_streak = 0
                        print(f"[INFO] ReID recovery: selected_id -> {best_tid} (sim={best_sim:.2f})")

        # Anzeige
        cv2.imshow(self.WINDOW_NAME, rgb)
        cv2.waitKey(1)

        return out_bbox, out_id
