# person_tracker.py
import cv2
import numpy as np
from ultralytics import YOLO

class HumanTracker:
    """
    YOLOv8 + ByteTrack Personentracker mit Mausklick-Auswahl.
    Methode .track(frame) liefert (bbox, track_id) der ausgewählten Person.
    """

    def __init__(self, model_path='runs/detect/yolov8_person_coco10/weights/best.pt'):
        # YOLOv8 Modell laden
        self.model = YOLO(model_path)
        self.selected_id = None
        self.latest_boxes = []

        # Fenster + Click-Handler initialisieren
        cv2.namedWindow('Tracked')
        cv2.setMouseCallback('Tracked', self._click_handler)

    def _click_handler(self, event, x, y, flags, param=None):
        # Beim Linksklick prüfen, in welcher Box der Klick liegt
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(box.id[0])
                    print(f"[INFO] Selected track ID: {self.selected_id}")
                    break

    def track(self, rgb: np.ndarray) -> tuple:
        """
        Führt Detection + Tracking durch, zeichnet alle Boxen in 'Tracked'.
        Args:
            frame_bgr: OpenCV-BGR-Image
        Returns:
            bbox: [x, y, w, h] der selektierten Person oder None
            track_id: int ID oder None
        """
        # RGB für YOLO
        results = self.model.track(rgb, conf=0.25, tracker="bytetrack.yaml", persist=True)

        self.latest_boxes = []
        bbox_out = None
        id_out = None

        # Jede erkannte Box verarbeiten
        for r in results:
            self.latest_boxes = r.boxes
            for box in self.latest_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1

                # Ist das die ausgewählte ID?
                if self.selected_id is not None and tid == self.selected_id:
                    bbox_out = [x1, y1, x2 - x1, y2 - y1]
                    id_out = tid

                # Zeichnen
                color = (0, 0, 255) if tid == self.selected_id else (0, 255, 0)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    rgb,
                    f"ID {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # Anzeige aktualisieren
        cv2.imshow('Tracked', rgb)
        cv2.waitKey(1)

        return bbox_out, id_out
