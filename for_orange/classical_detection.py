import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.optimize import curve_fit

from cfg import ClassicalCfg

class ClassicalDetection():

    def __init__(self, cfg = None):

        self.cfg = cfg or ClassicalCfg()

    def verify(self, image, grouped_data, img_number):

        self.image = image.copy()

        lanes = []
        bands = []

        for module in grouped_data[img_number]:
            for key, value in module.items():

                if key == 'zero':
                    continue

                elif key == 'line':
                    # Проверяем, что line не None
                    if value is not None:
                        bbox, conf = value[0], value[1]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[0] + bbox[2])
                        y2 = int(bbox[1] + bbox[3])

                        lanes.append({
                            "id": tuple(bbox),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "conf": conf  # Добавим conf для полноты
                        })

                elif key == 'strips':  # Обратите внимание: 'strips' (множественное число)
                    # Перебираем все strips
                    for strip_data in value:
                        bbox, conf = strip_data[0], strip_data[1]
                        # Для strip вычисляем центр по Y
                        strip_center_y = int(bbox[1] + bbox[3] // 2)

                        bands.append({
                            "lane_id": tuple(bbox),
                            "y": strip_center_y,
                            "ml_conf": conf
                        })

        # lanes : [{"id": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
        # bands : [{"lane_id": int, "y_in_lane": int}, ...]

        by_lane = {}

        for b in bands:
            by_lane.setdefault(b["lane_id"], []).append(b)

        all_results = []

        for lane in lanes:
            signal, noise, y_off = self._profile(lane)
            ml_pts = by_lane.get(lane["id"], [])
            verified, missed = self._verify_lane(signal, noise, y_off,
                                                  ml_pts, lane["id"])
            all_results.extend(verified + missed)

        return all_results

    def _profile(self, line: dict):

        cfg = self.cfg

        x1, y1, x2, y2 = int(line["x1"]), int(line["y1"]), int(line["x2"]), int(line["y2"])

        h, w = self.image.shape

        strip = self.image[y1:y2, x1:x2]
        tw = strip.shape[1]
        th = strip.shape[0]

        strip = cv2.createCLAHE(cfg.CLAHE_CLIP, (th, tw)).apply(strip)

        raw = strip.mean(axis=1).astype(float)

        sg = savgol_filter(raw, cfg.SG_WINDOW, cfg.SG_POLYORDER)

        bg = uniform_filter(minimum_filter(sg, cfg.RB_RADIUS), cfg.RB_RADIUS)
        signal = np.clip(sg - bg, 0, None)

        noise = max(0.5, np.median(np.abs(np.diff(signal))) / 0.6745)

        return signal, noise, y1


    def _verify_lane(self, signal, noise, y_off, ml_points, lane_id):

        cfg = self.cfg

        N = len(signal)

        cls_peaks, _ = find_peaks(
            signal,
            prominence = noise * 2.5,
            height     = noise * cfg.SNR_MIN,
            distance   = 8,
        )

        used = set()
        results = []

        for pt in sorted(ml_points, key=lambda p: p["y"]):
            y_rel = min(N - 1, int(pt["y"]) - y_off)

            cands = [p for p in cls_peaks if abs(p - y_rel) <= cfg.SHIFT_TOL and p not in used]

            if cands:
                best = min(cands, key=lambda p: abs(p - y_rel))
                used.add(best)
                delta = abs(best - y_rel)
                snr = float(signal[best] / noise)
                r2 = self._fit_r2(signal, best)
                pos_n = max(0.0, 1.0 - delta / cfg.SHIFT_TOL)

            else:
                best   = y_rel
                delta  = 0
                snr    = float(signal[y_rel] / noise)
                r2     = self._fit_r2(signal, y_rel)
                pos_n  = 0.0

            cls_s = self._score(snr, r2, pos_n)
            joint = cfg.ALPHA * pt["ml_conf"] + (1 - cfg.ALPHA) * cls_s

            results.append({
                "lane_id"        : lane_id,
                "ml_y"           : int(pt["y"]),
                "ml_conf"        : pt["ml_conf"],
                "classical_y"    : int(best + y_off),
                "classical_score": round(float(cls_s), 3),
            })

        missed = []

        for p in cls_peaks:
            if p in used:
                continue
            snr   = float(signal[p] / noise)
            r2    = self._fit_r2(signal, p)
            cls_s = self._score(snr, r2, 1.0)
            missed.append({
                "lane_id"        : lane_id,
                "ml_y"           : np.nan,
                "ml_conf"        : np.nan,
                "classical_y"    : int(p + y_off),
                "classical_score": round(float(cls_s), 3),
            })

        return results, missed


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
            yf = popt[0] * np.exp(-((x - popt[1]) ** 2) / (2 * popt[2] ** 2))
            ss_tot = np.sum((y - y.mean()) ** 2)
            return float(max(0.0, 1 - np.sum((y - yf) ** 2) / ss_tot)) if ss_tot > 0 else 0.0
        except Exception:
            return 0.0

    def _score(self, snr, r2, pos_norm):
        return 0.4 * min(1.0, snr / 10.0) + 0.4 * r2 + 0.2 * pos_norm