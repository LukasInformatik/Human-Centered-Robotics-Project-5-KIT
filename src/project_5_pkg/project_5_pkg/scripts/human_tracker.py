# person_tracker.py
import cv2, os
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine

class HumanTracker:
    """
    YOLOv8 + ByteTrack Personentracker mit Mausklick-Auswahl.
    Nutzt ReID-Feature-Extraction, um die Bounding Box der selektierten Person wiederzufinden,
    wenn BoT-SORT die ID verliert. Methode .track(frame) liefert (bbox, track_id).
    """

    def __init__(self):
        package_share = get_package_share_directory('project_5_pkg')
        model_path = os.path.join(package_share, 'models', 'yolov8n_80_epochs.pt')
        self.model = YOLO(model_path)
        self.selected_id = None
        self.latest_boxes = []

        # ReID-Feature-Extractor
        self.reid_extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=None,
            device='cpu'
        )
        self.selected_feature = None
        self.feature_threshold = 0.6

        cv2.namedWindow('Tracked')
        cv2.setMouseCallback('Tracked', self._click_handler)

    def _click_handler(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param.get('frame') if param else None
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(box.id[0])
                    print(f"[INFO] Selected track ID: {self.selected_id}")
                    if frame is not None:
                        self.selected_feature = self._extract_feature(frame, (x1, y1, x2, y2))
                    break

    def _extract_feature(self, frame: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        patch = frame[y1:y2, x1:x2]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        feat = self.reid_extractor(patch_rgb)
        return feat[0]

    def _recover_bbox(self, frame: np.ndarray):
        # ReID-basiertes Matching: Rückgabe der Bounding Box der selektierten Person
        best_sim = 0.0
        best_box = None
        best_id = None
        for box in self.latest_boxes:
            # nur neue Boxen (alle IDs, denn BoT-SORT hat selektierte ID verloren)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feat = self._extract_feature(frame, (x1, y1, x2, y2))
            sim = 1 - cosine(self.selected_feature, feat)
            if sim > self.feature_threshold and sim > best_sim:
                best_sim = sim
                best_box = (x1, y1, x2, y2)
                best_id = int(box.id[0])
        if best_box:
            print(f"[INFO] ReID recovered bbox for ID {self.selected_id}")
        return best_box, best_id

    def track(self, rgb: np.ndarray) -> tuple:
        results = self.model.track(rgb, conf=0.25, tracker="botsort.yaml", persist=True)
        self.latest_boxes = []
        bbox_out, id_out = None, None
        frame_copy = rgb.copy()

        for r in results:
            self.latest_boxes = r.boxes
            visible_ids = [int(b.id[0]) for b in self.latest_boxes if b.id is not None]

            # Wenn selektierte ID verloren und Feature vorhanden: bbox recovern
            if self.selected_id is not None \
               and self.selected_id not in visible_ids \
               and self.selected_feature is not None:
                rbox, rid = self._recover_bbox(frame_copy)
                if rbox is not None:
                    x1, y1, x2, y2 = rbox
                    bbox_out = [x1, y1, x2-x1, y2-y1]
                    self.selected_id = rid
                    
                    # behalten selected_feature unverändert oder updaten?
                    #self.selected_feature = self._extract_feature(frame_copy, (x1, y1, x2, y2))

            # Zeichnen und Ausgabe aktualisieren
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1

                if tid == self.selected_id:
                    bbox_out = [x1, y1, x2-x1, y2-y1]
                    id_out = tid
                    self.selected_feature = self._extract_feature(frame_copy, (x1, y1, x2, y2))

                color = (0,0,255) if tid == self.selected_id else (0,255,0)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb, f"ID {tid}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Anzeige aktualisieren
        cv2.imshow('Tracked', rgb)
        cv2.setMouseCallback('Tracked', self._click_handler, param={'frame': frame_copy})
        cv2.waitKey(1)

        return bbox_out, id_out
