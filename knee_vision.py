"""
knee_vision.py - camera helpers for ArthroKNEE (needs: opencv-python, numpy, scipy)

QR:      make_qr_png(text), decode_qr(img_bgr)
Camera:  detect_markers(img_bgr), knee_flexion_deg(pts), annotate(...), analyze_video(path)
Live:    live_stream(source, max_seconds) -> yields frames + angles; analyze_series(angles, fps)

Camera method = coloured stickers (side view of the leg):
    HIP = GREEN     KNEE = YELLOW     ANKLE = BLUE
Knee flexion = 180 - (angle hip-knee-ankle).  0 deg = straight leg.
"""
import sys
import time
import cv2
import numpy as np
from scipy.signal import find_peaks

HSV_RANGES = {
    "hip": [((40, 90, 60), (85, 255, 255))],       # green
    "knee": [((20, 100, 100), (35, 255, 255))],    # yellow
    "ankle": [((95, 100, 60), (130, 255, 255))],   # blue
}
DRAW_BGR = {"hip": (0, 200, 0), "knee": (0, 220, 255), "ankle": (255, 80, 0)}


# ---------------------------------------------------------------- helpers
def bytes_to_bgr(data):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- QR
def make_qr_png(text, scale=8, border=40):
    qr = cv2.QRCodeEncoder.create().encode(text)
    if qr is None:
        raise ValueError("QR encoding failed")
    big = cv2.resize(qr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    big = cv2.copyMakeBorder(big, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", big)
    return buf.tobytes()


def decode_qr(img_bgr):
    det = cv2.QRCodeDetector()
    val, _, _ = det.detectAndDecode(img_bgr)
    if val:
        return val
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    val, _, _ = det.detectAndDecode(big)
    return val or None


# ---------------------------------------------------------------- markers
def _centroid_of_largest(hsv, ranges, min_area):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > best_area:
            best, best_area = c, a
    if best is None or best_area < min_area:
        return None
    m = cv2.moments(best)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_markers(img_bgr):
    """Return {'hip':(x,y),'knee':(x,y),'ankle':(x,y)} or None if any is missing."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    min_area = max(25.0, 0.0002 * h * w)
    pts = {}
    for joint, ranges in HSV_RANGES.items():
        p = _centroid_of_largest(hsv, ranges, min_area)
        if p is None:
            return None
        pts[joint] = p
    return pts


def knee_flexion_deg(pts):
    hip, knee, ankle = (np.array(pts[k], float) for k in ("hip", "knee", "ankle"))
    v1, v2 = hip - knee, ankle - knee
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    interior = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return float(180.0 - interior)


def annotate(img_bgr, pts, angle):
    out = img_bgr.copy()
    order = ["hip", "knee", "ankle"]
    for a, b in zip(order[:-1], order[1:]):
        cv2.line(out, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                 (255, 255, 255), 3)
    for j in order:
        cv2.circle(out, tuple(int(v) for v in pts[j]), 11, DRAW_BGR[j], -1)
        cv2.circle(out, tuple(int(v) for v in pts[j]), 11, (255, 255, 255), 2)
    kx, ky = int(pts["knee"][0]), int(pts["knee"][1])
    txt = f"{angle:.0f} deg"
    cv2.putText(out, txt, (kx + 16, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(out, txt, (kx + 16, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------- video
def analyze_video(path, max_seconds=60, target_w=640):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 30.0
    angles, best_a, best_img = [], -1.0, None
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n > max_seconds * fps:
            break
        h, w = fr.shape[:2]
        if w > target_w:
            fr = cv2.resize(fr, (target_w, int(h * target_w / w)))
        pts = detect_markers(fr)
        if pts is None:
            angles.append(np.nan)
            continue
        a = knee_flexion_deg(pts)
        angles.append(a)
        if a > best_a:
            best_a, best_img = a, annotate(fr, pts, a)
    cap.release()

    return analyze_series(np.array(angles, float), fps, best_img)


def analyze_series(ang, fps, best_img=None):
    """Peak flexion + stride-time variability from a knee-angle time series (NaN = no detection)."""
    ang = np.asarray(ang, float)
    valid = ~np.isnan(ang) if len(ang) else np.array([], bool)
    if len(ang) == 0 or valid.sum() < max(10, 0.2 * len(ang)):
        return dict(ok=False, frames=len(ang), fps=fps,
                    detected_pct=100.0 * valid.mean() if len(ang) else 0.0)

    idx = np.arange(len(ang))
    filled = np.interp(idx, idx[valid], ang[valid])
    k = max(3, int(fps * 0.1))
    k += (k + 1) % 2                      # make odd
    smooth = np.convolve(filled, np.ones(k) / k, mode="same")
    peak = float(np.percentile(smooth[valid], 95))
    prom = max(8.0, 0.25 * float(smooth.max() - smooth.min()))
    peaks, _ = find_peaks(smooth, prominence=prom, distance=max(1, int(0.4 * fps)))
    cv = None
    if len(peaks) >= 4:
        iv = np.diff(peaks) / fps
        cv = float(np.std(iv) / np.mean(iv) * 100.0)
    cadence = None                                   # steps/min (2 steps per stride of this leg)
    if len(peaks) >= 3:
        span = (peaks[-1] - peaks[0]) / fps
        if span > 0:
            cadence = float(2 * (len(peaks) - 1) / span * 60.0)
    return dict(ok=True, frames=len(ang), fps=fps, detected_pct=100.0 * valid.mean(),
                peak_flexion=peak, n_peaks=int(len(peaks)), stride_cv=cv, cadence=cadence,
                series=smooth, best_frame=best_img)


# ---------------------------------------------------------------- live camera
def _open_capture(source):
    """source = camera number (int) or a video file path (str, used for testing)."""
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        cap = None
        if sys.platform == "win32":                       # DirectShow is more reliable on Windows
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
    return cap if cap.isOpened() else None


def live_stream(source=0, max_seconds=20, target_w=480):
    """Generator: reads the webcam, tracks the 3 stickers, yields one dict per frame:
         frame (annotated BGR), angle, t (seconds), fps, angles (list so far), best_frame
    Closes the camera when finished or when the generator is closed."""
    cap = _open_capture(source)
    if cap is None:
        raise ValueError("Could not open camera")
    file_fps = cap.get(cv2.CAP_PROP_FPS) if isinstance(source, str) else None
    angles, best_a, best_img = [], -1.0, None
    t0, n = time.perf_counter(), 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            now = (n / file_fps) if file_fps else (time.perf_counter() - t0)
            if now > max_seconds:
                break
            h, w = fr.shape[:2]
            if w > target_w:
                fr = cv2.resize(fr, (target_w, int(h * target_w / w)))
            pts = detect_markers(fr)
            if pts is None:
                a, shown = float("nan"), fr
            else:
                a = knee_flexion_deg(pts)
                shown = annotate(fr, pts, a)
                if a > best_a:
                    best_a, best_img = a, shown
            angles.append(a)
            yield dict(frame=shown, angle=a, t=now, fps=(n / now if now > 0 else 0.0),
                       angles=angles, best_frame=best_img)
    finally:
        cap.release()
