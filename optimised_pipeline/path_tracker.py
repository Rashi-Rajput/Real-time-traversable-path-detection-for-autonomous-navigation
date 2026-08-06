import cv2
import numpy as np


class KalmanPoint:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.age = 0
        self.max_age = 10
        self.matched = False

    def predict(self):
        return self.kf.predict()

    def correct(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        self.age = 0
        self.matched = True

    def get_state(self):
        state = self.kf.statePost.flatten()
        return float(state[0]), float(state[1])


class PathKalmanTracker:
    def __init__(self, max_y_gap=15):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 10
        self.max_y_gap = max_y_gap

    def update(self, raw_points):
        if not raw_points:
            for track in self.tracks.values():
                track.age += 1
            self.tracks = {tid: t for tid, t in self.tracks.items() if t.age < t.max_age}
            return []

        for track in self.tracks.values():
            track.predict()
            track.matched = False

        matched_measurements = set()

        track_list = sorted(self.tracks.items(), key=lambda item: item[1].get_state()[1])

        for tid, track in track_list:
            ty = track.get_state()[1]
            best_m = None
            best_d = float("inf")

            for i, (mx, my) in enumerate(raw_points):
                if i in matched_measurements:
                    continue
                d = abs(ty - my)
                if d < best_d and d < self.max_y_gap:
                    best_d = d
                    best_m = i

            if best_m is not None:
                mx, my = raw_points[best_m]
                track.correct(mx, my)
                track.matched = True
                matched_measurements.add(best_m)

        for track in self.tracks.values():
            if not track.matched:
                track.age += 1

        for i, (x, y) in enumerate(raw_points):
            if i not in matched_measurements:
                new_track = KalmanPoint(x, y)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t.age < self.max_age}

        smoothed = []
        for track in sorted(self.tracks.values(), key=lambda t: t.get_state()[1], reverse=True):
            sx, sy = track.get_state()
            smoothed.append((int(sx), int(sy)))

        return smoothed
