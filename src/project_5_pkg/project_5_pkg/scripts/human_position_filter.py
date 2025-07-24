# human_position_kf.py

from filterpy.kalman import KalmanFilter
import numpy as np

class HumanPositionKalmanFilter:
    """
    3D Constant‑Velocity Kalman‑Filter für menschliche Positionen.
    State x = [px, py, pz, vx, vy, vz]^T
    Measurement z = [px, py, pz]^T
    """

    def __init__(self,
                 initial_dt: float = 1/30.0,
                 process_noise: float = 1e-4,
                 measurement_noise: float = 0.01):
        # 1) Filter erstellen
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # 2) State‑Transition‑Matrix F initialisieren
        self.set_dt(initial_dt)

        # 3) Messmatrix H: wir messen nur Position
        self.kf.H = np.hstack([np.eye(3), np.zeros((3,3))])

        # 4) Rauschkovarianzen
        self.kf.R *= measurement_noise    # Messrauschen
        self.kf.Q *= process_noise        # Prozessrauschen
        self.kf.P *= 1.0                  # Start‑Kovarianz
        self.kf.x = np.zeros((6,1))       # Startzustand

        # für dynamisches dt
        self.last_time = None

    def set_dt(self, dt: float):
        """
        Aktualisiert die F‑Matrix für neuen Zeitschritt dt.
        """
        F = np.eye(6)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt
        self.kf.F = F

    def update(self,
               position_meas: np.ndarray,
               timestamp: float = None) -> np.ndarray:
        """
        Führt einen Filter‑Schritt durch.
        :param position_meas: np.array([px,py,pz])
        :param timestamp: Zeitstempel in Sekunden (float), optional
        :return: gefilterte Position np.array([px,py,pz])
        """
        # dynamisches dt, falls Zeitstempel übergeben
        if timestamp is not None:
            if self.last_time is not None:
                dt = timestamp - self.last_time
                self.set_dt(dt)
            self.last_time = timestamp

        # 1) Predict
        self.kf.predict()
        # 2) Update
        self.kf.update(position_meas)

        # zurückgeben: geglättete Position
        return self.kf.x[:3].flatten()
