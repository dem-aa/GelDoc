import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.optimize import curve_fit

from pre_and_post_proc import PostProc
from cfg import ClassicalCfg


class ClassicalDetect():

    def __init__(self, cfg=None):
        self.cfg = cfg or ClassicalCfg()

    def detect(self, image: np.ndarray, results: dict, cords: tuple):
        self.image = image.copy()

        lanes, _ = self._parse_yolo_results(results)
        lanes = PostProc.check_in_gel(image, lanes, cords)

        all_results = []
        for lane in lanes:
            signal, noise, y_off = self._profile(lane)
            bands = self._detect_bands(signal, noise, y_off, lane["id"])
            all_results.extend(bands)

        return all_results

    def _detect_bands(self, signal, noise, y_off, lane_id):
        cfg = self.cfg

        peaks, _ = find_peaks(
            signal,
            prominence=noise * 8.0,
            height=noise * cfg.SNR_MIN,
            distance=10,
        )

        results = []
        for p in peaks:
            snr   = float(signal[p] / noise)
            r2    = self._fit_r2(signal, p)
            score = self._score(snr, r2, 1.0)
            results.append({
                "lane_id"        : lane_id,
                "ml_y"           : None,
                "ml_conf"        : None,
                "classical_y"    : int(p + y_off),
                "classical_score": round(float(score), 3),
            })

        return results

    def _parse_yolo_results(self, results: dict):
        lanes, bands = [], []

        for key, value in results.items():
            if key == 0 or value is None:
                continue

            if key == 2:
                for x1, y1, x2, y2, conf in value:
                    lanes.append({
                        "id": (x1, y1, x2, y2),
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "conf": conf,
                    })

        return lanes, bands

    def _profile(self, line: dict):
        cfg = self.cfg
        h, w = self.image.shape[:2]
        sx, sy = w / 1280, h / 1280

        x1 = int(line["x1"] * sx)
        y1 = int(line["y1"] * sy)
        x2 = int(line["x2"] * sx)
        y2 = int(line["y2"] * sy)

        strip = self.image[y1:y2, x1:x2]
        raw   = strip.mean(axis=1).astype(float)

        if len(raw) < cfg.SG_WINDOW:
            return np.zeros(1), 1.0, y1

        sg     = savgol_filter(raw, cfg.SG_WINDOW, cfg.SG_POLYORDER)
        bg     = uniform_filter(minimum_filter(sg, cfg.RB_RADIUS), cfg.RB_RADIUS)
        signal = np.clip(sg - bg, 0, None)

        mad   = np.median(np.abs(np.diff(signal))) / 0.6745
        noise = max(mad, signal.max() * 0.01)

        return signal, noise, y1

    def _fit_r2(self, signal, center, window=15):
        lo, hi = max(0, center - window), min(len(signal), center + window)
        if hi - lo < 4:
            return 0.0
        x, y = np.arange(lo, hi, dtype=float), signal[lo:hi]
        try:
            popt, _ = curve_fit(
                lambda x, A, mu, s: A * np.exp(-((x - mu) ** 2) / (2 * s ** 2)),
                x, y,
                p0=[float(signal[center]), float(center), 4.0],
                bounds=([0, lo, 0.5], [np.inf, hi, 40.0]),
                maxfev=3000,
            )
            yf     = popt[0] * np.exp(-((x - popt[1]) ** 2) / (2 * popt[2] ** 2))
            ss_tot = np.sum((y - y.mean()) ** 2)
            return float(max(0.0, 1 - np.sum((y - yf) ** 2) / ss_tot)) if ss_tot > 0 else 0.0
        except Exception:
            return 0.0

    def _score(self, snr, r2, pos_norm):
        return 0.4 * min(1.0, snr / 10.0) + 0.4 * r2 + 0.2 * pos_norm
