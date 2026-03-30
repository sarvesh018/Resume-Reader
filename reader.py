import os
import glob
import re
import cv2
import csv
import shutil
import tkinter as tk
from tkinter import filedialog
import numpy as np
from skimage.metrics import structural_similarity as ssim

# =========================
# >>>>> CONFIG (ALGORITHM THRESHOLDS ONLY) <<<<<
# (No input folder hard-coded anymore; handled by GUI.)
# =========================

# -------- Graph presence thresholds --------
DARK_RATIO_MIN           = 0.33
EDGE_DENSITY_MIN         = 0.018
GRID_GREEN_MIN           = 0.001
HUE_EXCLUDE_GREEN        = (55, 80)    # OpenCV hue 0..179

# -------- Axis & duplicates (UNCHANGED) --------a
AXIS_BAND_HEIGHT_RATIO   = 0.13
SSIM_WORKING_WIDTH       = 384
SSIM_THRESHOLD_DUP       = 0.93

# -------- v3+ trace extraction (NO OCR) --------
REQUIRED_TRACES          = 2

# Preprocessing & edge
VERT_LINE_KERNEL_H       = 11
MEDIAN_BLUR_KSIZE        = 3
CANNY_LOWER_FACTOR       = 0.66
CANNY_UPPER_EXTRA        = 30
DILATE_KERNEL_SIZE       = (3, 3)

# Color/neutral candidate masks
S_MIN_COLORFUL           = 60  # floor, we'll clamp to >=50 in code
V_MIN_COLORFUL           = 55
S_MAX_NEUTRAL            = 60
V_MIN_NEUTRAL            = 170

# Component validation
MIN_TRACE_PIXELS         = 300
MIN_COLUMN_COVERAGE      = 0.08
MAX_STROKE_WIDTH_PX      = 7
MIN_CONTRAST_DELTA       = 5
VLINE_ASPECT_MIN         = 4.0

# Merge fragments
Y_MERGE_TOL_PX           = 14
X_GAP_TOL_PX_MIN         = 12
X_GAP_TOL_FRAC_W         = 1/90

# Hough horizontal line detection (signal leg)
HOUGH_MIN_LINE_LEN_FRAC  = 0.12
HOUGH_MAX_GAP_PX         = 8
HOUGH_THETA_DEG_TOL      = 5
HOUGH_ACCUM_THRESHOLD    = 30

# --- Horizontal baselines (panel references) suppression ---
# Stage-1a: Hough-XL (extra-long line killer), color-agnostic
HXL_MIN_LINE_LEN_FRAC    = 0.95   # >=95% of width
HXL_MAX_GAP_PX           = 4
HXL_THETA_DEG_TOL        = 3
HXL_BAND_RADIUS_PX       = 2

# Stage-1b: neutral-row suppression (secondary)
HREF_NEUTRAL_S_MAX       = 70
HREF_NEUTRAL_V_MIN       = 160
HREF_NEUTRAL_ROW_FRAC    = 0.65
HREF_BAND_RADIUS_PX      = 2

# --- Post-filter in final traces ---
REF_COL_COV_DROP         = 0.90   # near full width
REF_THICK_MAX_DROP       = 4.0    # ultra-thin
REF_NEUTRAL_FRAC_MIN     = 0.70   # mostly neutral region
BASELINE_OVERLAP_FRAC_MIN= 0.60   # overlaps baseline rows

# Vertical-edge test (reject pure horizontals)
VERT_EDGE_BIN_THR        = 30     # Sobel-Y threshold
VERT_EDGE_MIN_DENSITY    = 0.003  # min fraction of strong vertical edges in bbox

# =========================
# Utilities (with Windows-safe debug names)
# =========================
WINDOWS_FORBIDDEN = r'<>:"/\|?*'

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _cv2_imwrite_safe(full_path, bgr):
    try:
        ok = cv2.imwrite(full_path, bgr)
        if not ok:
            print(f"[WARN] Failed to write: {full_path}")
        return ok
    except Exception as e:
        print(f"[WARN] Exception writing: {full_path} -> {e}")
        return False

def safe_name(name: str) -> str:
    name = name.rstrip(" .")
    for ch in WINDOWS_FORBIDDEN:
        name = name.replace(ch, "_")
    name = re.sub(r"\s+", " ", name).strip()
    return name or "unnamed"

def imsave_rgb(path, rgb):
    if not path:
        return
    ensure_dir(os.path.dirname(path))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _cv2_imwrite_safe(path, bgr)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def text_binarize(img_gray):
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - bw

def extract_numeric_index(file_name):
    m = re.search(r'(\d+)(?:\D*)$', os.path.splitext(file_name)[0])
    return int(m.group(1)) if m else None

# =========================
# Legend/Graph split (brightness gradient)
# =========================
def split_legend_and_graph(rgb, min_graph_w_ratio=0.55):
    h, w, _ = rgb.shape
    gray = to_gray(rgb)
    col_m = gray.mean(axis=0)
    k = max(5, w // 300)
    col_s = cv2.blur(col_m.reshape(1, -1), (1, k)).flatten()

    left = int(0.05 * w)
    right = int(0.35 * w)
    left = max(0, left); right = min(w-2, right)

    grad = np.diff(col_s.astype(np.float32))
    search = grad[left:right]
    if len(search) == 0:
        split_x = int(0.30 * w)
    else:
        split_x = int(np.argmin(search)) + left
    split_x = np.clip(split_x, int(0.10 * w), int(0.40 * w))

    legend = rgb[:, :split_x]
    graph  = rgb[:, split_x:]

    if graph.shape[1] < int(min_graph_w_ratio * w):
        split_x = int(0.30 * w)
        legend = rgb[:, :split_x]
        graph  = rgb[:, split_x:]

    return legend, graph, split_x

# =========================
# Stage 1: Graph window presence (unchanged)
# =========================
def has_graph_window(graph_rgb):
    if graph_rgb is None or graph_rgb.size == 0:
        return False, {"reason": "empty_graph_roi"}

    gray = to_gray(graph_rgb)
    dark_ratio = (gray < 60).mean()
    edge_density = cv2.Canny(gray, 40, 120).mean()

    hsv = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    green_ratio = ((H >= HUE_EXCLUDE_GREEN[0]) & (H <= HUE_EXCLUDE_GREEN[1]) & (S >= 60) & (V >= 40)).mean()

    present = (dark_ratio > DARK_RATIO_MIN and edge_density > EDGE_DENSITY_MIN) or \
              (dark_ratio > DARK_RATIO_MIN and green_ratio > GRID_GREEN_MIN)

    info = dict(dark_ratio=float(dark_ratio),
                edge_density=float(edge_density),
                green_ratio=float(green_ratio),
                present=bool(present))
    if not present:
        info["reason"] = "no_graph_window"
    return present, info

# =========================
# Baseline suppression (Hough-XL and neutral rows)
# =========================
def _hough_xl_rows(gray):
    h, w = gray.shape
    e = cv2.Canny(gray, 40, 120)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), iterations=1)

    min_len = int(HXL_MIN_LINE_LEN_FRAC * w)
    lines = cv2.HoughLinesP(e, 1, np.pi/180, threshold=60,
                            minLineLength=min_len, maxLineGap=HXL_MAX_GAP_PX)
    mask = np.zeros(h, dtype=bool)
    if lines is None:
        return mask

    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1 + 1e-6)))
        if ang <= HXL_THETA_DEG_TOL:
            yc = int(round((y1 + y2) / 2))
            mask[max(0, yc - HXL_BAND_RADIUS_PX):min(h, yc + HXL_BAND_RADIUS_PX + 1)] = True
    return mask

def _neutral_rows(work_rgb):
    h, w, _ = work_rgb.shape
    hsv = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2HSV)
    _, S, V = cv2.split(hsv)
    neutral = ((S <= HREF_NEUTRAL_S_MAX) & (V >= HREF_NEUTRAL_V_MIN)).astype(np.uint8)
    row_neutral_frac = neutral.mean(axis=1)
    mask = (row_neutral_frac >= HREF_NEUTRAL_ROW_FRAC)
    return mask

def suppress_baselines(work_rgb):
    gray = to_gray(work_rgb)
    h, w = gray.shape

    # Hough-XL
    long_mask = _hough_xl_rows(gray)
    if long_mask.any():
        ys = np.where(long_mask)[0]
        for r in ys:
            y1 = max(0, r - HXL_BAND_RADIUS_PX)
            y2 = min(h - 1, r + HXL_BAND_RADIUS_PX)
            work_rgb[y1:y2+1, :, :] = 0

    # Neutral row
    nr_mask = _neutral_rows(work_rgb)
    if nr_mask.any():
        ys = np.where(nr_mask)[0]
        for r in ys:
            y1 = max(0, r - HREF_BAND_RADIUS_PX)
            y2 = min(h - 1, r + HREF_BAND_RADIUS_PX)
            work_rgb[y1:y2+1, :, :] = 0

    return work_rgb

# =========================
# v3: Grid suppression + Baseline suppression + Edge map
# =========================
def suppress_grid_and_prepare(graph_rgb):
    hsv = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    grid_mask = ((H >= HUE_EXCLUDE_GREEN[0]) & (H <= HUE_EXCLUDE_GREEN[1]) & (S >= 50) & (V >= 30)).astype(np.uint8) * 255
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(VERT_LINE_KERNEL_H)))
    grid_vert = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, vkernel, iterations=1)

    work = graph_rgb.copy()
    work[grid_vert > 0] = (0, 0, 0)

    if MEDIAN_BLUR_KSIZE >= 3 and MEDIAN_BLUR_KSIZE % 2 == 1:
        work = cv2.medianBlur(work, MEDIAN_BLUR_KSIZE)

    # Remove baseline rows aggressively (Hough-XL + neutral)
    work = suppress_baselines(work)

    gray = to_gray(work)
    return work, gray

def build_edge_map(gray):
    v = np.median(gray)
    lower = max(10, int(CANNY_LOWER_FACTOR * v))
    upper = min(255, int(v * 1.33) + CANNY_UPPER_EXTRA)
    edges = cv2.Canny(gray, lower, upper)
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL_SIZE)
    edges = cv2.dilate(edges, dk, iterations=1)
    return edges

# =========================
# v3: Component extraction + validation + merging
# =========================
def candidate_mask(work_rgb, edges):
    hsv = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2HSV)
    _, S, V = cv2.split(hsv)
    colorful  = (S >= max(50, S_MIN_COLORFUL)) & (V >= max(50, V_MIN_COLORFUL))
    mask_color = colorful.astype(np.uint8)
    neutral = ((S <= S_MAX_NEUTRAL) & (V >= V_MIN_NEUTRAL)).astype(np.uint8)
    cand = (edges > 0).astype(np.uint8) | mask_color | neutral
    return cand

def estimate_contrast(gray, comp_mask):
    dil = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    ring = ((dil > 0) & (comp_mask == 0))
    inside_vals = gray[comp_mask > 0]
    ring_vals = gray[ring]
    if inside_vals.size == 0 or ring_vals.size == 0:
        return 0.0
    return float(abs(np.median(inside_vals) - np.median(ring_vals)))

def extract_components(binary_mask, gray):
    h, w = binary_mask.shape
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    comps = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < MIN_TRACE_PIXELS:
            continue
        if ww < 8 and (hh / max(1.0, ww)) >= VLINE_ASPECT_MIN:
            continue
        comp_mask = (labels == i).astype(np.uint8)
        col_active = (comp_mask.sum(axis=0) > 0)
        cov = float(col_active.mean())
        if cov < MIN_COLUMN_COVERAGE:
            continue
        active_cols = max(1, int(col_active.sum()))
        avg_thick = float(area) / active_cols
        if avg_thick > MAX_STROKE_WIDTH_PX:
            continue
        contrast = estimate_contrast(gray, comp_mask)
        if contrast < MIN_CONTRAST_DELTA:
            continue
        cx, cy = centroids[i]
        comps.append({
            "bbox": (int(x), int(y), int(ww), int(hh)),
            "area": int(area),
            "x1": int(x), "x2": int(x + ww - 1),
            "yc": float(cy),
            "col_cov": cov,
            "avg_thick": float(avg_thick),
            "contrast": float(contrast),
            "pixels": int(area)
        })
    return comps

def merge_components(comps, width):
    if not comps:
        return []
    comps = sorted(comps, key=lambda c: (int(round(c["yc"])), c["x1"]))
    used = [False] * len(comps)
    merged = []
    x_gap_tol = max(X_GAP_TOL_PX_MIN, int(X_GAP_TOL_FRAC_W * width))
    for i in range(len(comps)):
        if used[i]:
            continue
        cur = comps[i].copy()
        used[i] = True
        for j in range(i+1, len(comps)):
            if used[j]:
                continue
            cy_ok = abs(comps[j]["yc"] - cur["yc"]) <= Y_MERGE_TOL_PX
            gap = comps[j]["x1"] - cur["x2"]
            if cy_ok and gap <= x_gap_tol:
                cur["x1"] = min(cur["x1"], comps[j]["x1"])
                cur["x2"] = max(cur["x2"], comps[j]["x2"])
                cur["area"] += comps[j]["area"]
                cur["pixels"] += comps[j]["pixels"]
                y1 = min(cur["bbox"][1], comps[j]["bbox"][1])
                y2 = max(cur["bbox"][1] + cur["bbox"][3], comps[j]["bbox"][1] + comps[j]["bbox"][3])
                x1 = cur["x1"]; x2 = cur["x2"]
                cur["bbox"] = (x1, y1, x2 - x1 + 1, y2 - y1)
                used[j] = True
        cur["col_cov"] = (cur["x2"] - cur["x1"] + 1) / max(1.0, width)
        merged.append(cur)
    return merged

# =========================
# Hough horizontal lines (signal leg)
# =========================
def detect_horizontal_traces_with_hough(gray, axis_band_ratio=AXIS_BAND_HEIGHT_RATIO):
    h, w = gray.shape
    band_h = max(8, int(axis_band_ratio * h))
    roi = gray[:h - band_h, :]
    v = np.median(roi)
    lower = max(5, int(0.5 * v))
    upper = min(255, int(v * 1.2) + 20)
    e = cv2.Canny(roi, lower, upper)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), iterations=1)
    min_len = int(HOUGH_MIN_LINE_LEN_FRAC * w)
    lines = cv2.HoughLinesP(e, 1, np.pi/180, HOUGH_ACCUM_THRESHOLD,
                            minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP_PX)
    if lines is None:
        return []
    horiz = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        theta = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1 + 1e-6))))
        if theta <= HOUGH_THETA_DEG_TOL:
            horiz.append((x1, y1, x2, y2))
    if not horiz:
        return []
    horiz = sorted(horiz, key=lambda p: (int(round((p[1] + p[3]) / 2)), min(p[0], p[2])))
    Y_TOL = 6
    clusters = []
    for x1, y1, x2, y2 in horiz:
        yc = int(round((y1 + y2) / 2))
        xa, xb = min(x1, x2), max(x1, x2)
        if not clusters:
            clusters.append([yc, xa, xb, yc, 1]); continue
        cy, lx, rx, sumy, cnt = clusters[-1]
        if abs(yc - cy) <= Y_TOL and xa <= rx + HOUGH_MAX_GAP_PX:
            clusters[-1][1] = min(lx, xa); clusters[-1][2] = max(rx, xb)
            clusters[-1][3] += yc; clusters[-1][4] += 1
        else:
            clusters.append([yc, xa, xb, yc, 1])
    traces = []
    for cy, lx, rx, sumy, cnt in clusters:
        yc = sumy / cnt
        x1 = max(0, lx); x2 = min(w-1, rx)
        col_cov = (x2 - x1 + 1) / max(1.0, w)
        y_top = max(0, int(yc) - 3)
        y_bot = min(h - band_h - 1, int(yc) + 3)
        bbox = (x1, y_top, x2 - x1 + 1, y_bot - y_top + 1)
        area = bbox[2] * bbox[3]
        traces.append({
            "bbox": bbox, "area": int(area),
            "x1": int(x1), "x2": int(x2), "yc": float(yc),
            "col_cov": float(col_cov), "avg_thick": 3.0,
            "contrast": 10.0, "pixels": int(area)
        })
    return traces

# =========================
# Post-filter helpers
# =========================
def neutral_fraction_in_bbox(work_rgb, bbox, s_max=HREF_NEUTRAL_S_MAX, v_min=HREF_NEUTRAL_V_MIN):
    x, y, w, h = bbox
    H, W = work_rgb.shape[:2]
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    x2 = min(W, x + w); y2 = min(H, y + h)
    roi = work_rgb[y:y2, x:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    _, S, V = cv2.split(hsv)
    neutral = ((S <= s_max) & (V >= v_min)).mean()
    return float(neutral)

def baseline_overlap_frac(work_rgb, bbox):
    x, y, w, h = bbox
    H, W = work_rgb.shape[:2]
    y1 = max(0, int(y)); y2 = min(H, int(y + h))
    if y2 <= y1:
        return 0.0
    mask = _neutral_rows(work_rgb)
    rows = np.arange(y1, y2)
    if rows.size == 0:
        return 0.0
    return float(mask[rows].sum()) / float(rows.size)

def vertical_edge_density(gray, bbox):
    x, y, w, h = bbox
    H, W = gray.shape[:2]
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    x2 = min(W, x + w); y2 = min(H, y + h)
    roi = gray[y:y2, x:x2]
    if roi.size == 0:
        return 0.0
    sobel_y = cv2.Sobel(roi, cv2.CV_16S, 0, 1, ksize=3)
    v_edges = (np.abs(sobel_y) > VERT_EDGE_BIN_THR).mean()
    return float(v_edges)

# =========================
# Detect traces (component leg + Hough leg, union + light merge) + final filters
# =========================
def detect_traces_v3(graph_rgb):
    work, gray = suppress_grid_and_prepare(graph_rgb)
    edges = build_edge_map(gray)

    # Component leg
    cand = candidate_mask(work, edges)
    comps = extract_components(cand, gray)
    traces_comp = merge_components(comps, cand.shape[1])

    # Hough horizontal leg (signal)
    traces_hough = detect_horizontal_traces_with_hough(gray)

    # Union + light merge
    all_traces = traces_comp + traces_hough
    if not all_traces:
        return []

    all_traces = sorted(all_traces, key=lambda t: (int(round(t["yc"])), t["x1"]))
    merged = []
    Y_TOL = 8
    for t in all_traces:
        if not merged:
            merged.append(t); continue
        m = merged[-1]
        same_y = abs(t["yc"] - m["yc"]) <= Y_TOL
        overlap = not (t["x1"] > m["x2"] or t["x2"] < m["x1"])
        if same_y and overlap:
            m["x1"] = min(m["x1"], t["x1"])
            m["x2"] = max(m["x2"], t["x2"])
            x1, x2 = m["x1"], m["x2"]
            y1 = min(m["bbox"][1], t["bbox"][1])
            y2 = max(m["bbox"][1] + m["bbox"][3], t["bbox"][1] + t["bbox"][3])
            m["bbox"] = (x1, y1, x2 - x1 + 1, y2 - y1)
            m["col_cov"] = (x2 - x1 + 1) / max(1.0, cand.shape[1])
            m["area"] = m["bbox"][2] * m["bbox"][3]
        else:
            merged.append(t)

    # Final: drop reference-like bands (any one of the discriminators can trigger drop)
    filtered = []
    for t in merged:
        bbox      = t.get("bbox", (0,0,1,1))
        col_cov   = float(t.get("col_cov", 0.0))
        thick     = float(t.get("avg_thick", 0.0))
        neutral_f = neutral_fraction_in_bbox(work, bbox)
        base_ovlp = baseline_overlap_frac(work, bbox)
        v_edge    = vertical_edge_density(gray, bbox)

        is_ultrathin_full = (col_cov >= REF_COL_COV_DROP and thick <= REF_THICK_MAX_DROP)
        looks_baseline    = (neutral_f >= REF_NEUTRAL_FRAC_MIN) or (base_ovlp >= BASELINE_OVERLAP_FRAC_MIN) or (v_edge < VERT_EDGE_MIN_DENSITY)

        if is_ultrathin_full and looks_baseline:
            continue
        filtered.append(t)

    return filtered

# =========================
# Axis band & duplicates (adjacent only)  >>> UNCHANGED <<<
# =========================
def crop_axis_band(graph_rgb):
    h, w, _ = graph_rgb.shape
    band_h = max(8, int(AXIS_BAND_HEIGHT_RATIO * h))
    return graph_rgb[h - band_h : h, :, :]

def axis_band_equal(graph_a_rgb, graph_b_rgb):
    a = crop_axis_band(graph_a_rgb)
    b = crop_axis_band(graph_b_rgb)
    if a.shape != b.shape:
        return False
    ga = text_binarize(to_gray(a))
    gb = text_binarize(to_gray(b))
    return np.array_equal(ga, gb)

def ssim_graph_above_axis(graph_a_rgb, graph_b_rgb):
    def top_strip(img):
        band_h = max(8, int(AXIS_BAND_HEIGHT_RATIO * img.shape[0]))
        top = img[:img.shape[0] - band_h, :, :]
        if top.shape[1] != SSIM_WORKING_WIDTH:
            scale = SSIM_WORKING_WIDTH / top.shape[1]
            top = cv2.resize(top, (SSIM_WORKING_WIDTH, int(top.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return to_gray(top)
    ga = top_strip(graph_a_rgb)
    gb = top_strip(graph_b_rgb)
    h = min(ga.shape[0], gb.shape[0])
    ga = ga[:h, :]
    gb = gb[:h, :]
    return float(ssim(ga, gb))

# =========================
# Per-file processing (unchanged core behavior)
# =========================
def process_file(path, debug_dir=""):
    rgb = imread_rgb(path)
    if rgb is None:
        return None

    legend, graph, _ = split_legend_and_graph(rgb)

    if debug_dir:
        unsafe_base = os.path.splitext(os.path.basename(path))[0]
        base = safe_name(unsafe_base)
        imsave_rgb(os.path.join(debug_dir, base, "legend.png"), legend)
        imsave_rgb(os.path.join(debug_dir, base, "graph.png"), graph)

    present, s1info = has_graph_window(graph)
    if not present:
        if debug_dir:
            unsafe_base = os.path.splitext(os.path.basename(path))[0]
            base = safe_name(unsafe_base)
            imsave_rgb(os.path.join(debug_dir, base, "traces_overlay.png"), graph)
        return {"label":"BAD", "reason":{"stage":"S1", **s1info}, "graph":graph}

    traces = detect_traces_v3(graph)
    valid_traces = len(traces)

    if debug_dir:
        unsafe_base = os.path.splitext(os.path.basename(path))[0]
        base = safe_name(unsafe_base)
        overlay = graph.copy()
        for t in traces:
            x, y, w, h = t["bbox"]
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w-1), int(y+h-1)), (255,255,0), 2)
        imsave_rgb(os.path.join(debug_dir, base, "traces_overlay.png"), overlay)

    if valid_traces < REQUIRED_TRACES:
        return {"label":"BAD",
                "reason":{"stage":"S2",
                          "reason":"insufficient_traces",
                          "valid_traces": valid_traces,
                          "traces": traces},
                "graph":graph}

    return {"label":"GOOD",
            "reason":{"stage":"OK","valid_traces": valid_traces,"traces": traces},
            "graph":graph}

# =========================
# Summary & CSV writers
# =========================
def write_summary_txt(path, results, dup_pairs, title="FINAL"):
    ensure_dir(os.path.dirname(path))
    total = len(results)
    bad_items = [(fp, r["reason"]) for fp, r in results.items() if r["label"] == "BAD"]
    good_items = [(fp, r["reason"]) for fp, r in results.items() if r["label"] == "GOOD"]

    s1_count = sum(1 for _, reason in bad_items if reason.get("stage") == "S1")
    s2_count = sum(1 for _, reason in bad_items if reason.get("stage") == "S2")
    bad_count = len(bad_items)
    good_count = len(good_items)
    dup_count  = len(dup_pairs)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY ===\n")
        f.write(f"Total PNGs       : {total}\n")
        f.write(f"BAD total        : {bad_count}\n")
        f.write(f"  - S1 (no graph): {s1_count}\n")
        f.write(f"  - S2 (traces)  : {s2_count}\n")
        f.write(f"GOOD total       : {good_count}\n")
        f.write(f"Duplicate pairs  : {dup_count}\n\n")

        f.write(f"=== BAD GRAPHS ({title}) ===\n")
        for fp, reason in sorted(bad_items, key=lambda x: os.path.basename(x[0]).lower()):
            base = os.path.basename(fp)
            stage = reason.get("stage", "")
            why   = reason.get("reason", "")
            details = []
            for k in ["dark_ratio","edge_density","green_ratio","valid_traces","traces"]:
                if k in reason:
                    details.append(f"{k}={reason[k]}")
            extra = (" | " + ", ".join(map(str,details))) if details else ""
            f.write(f"{base}  --  {stage}:{why}{extra}\n")

        f.write("\n=== DUPLICATES (adjacent only, exact axis) ===\n")
        if not dup_pairs:
            f.write("(none)\n")
        else:
            for (a, b, score) in dup_pairs:
                f.write(f"{os.path.basename(a)} == {os.path.basename(b)} (SSIM={score:.3f})\n")

        f.write("\n=== GOOD FILES (for reference) ===\n")
        for fp, _ in sorted(good_items, key=lambda x: os.path.basename(x[0]).lower()):
            f.write(f"{os.path.basename(fp)}\n")

def write_summary_csv(path, results, dup_pairs):
    ensure_dir(os.path.dirname(path))
    # Map dup membership
    dup_map = {}
    for a, b, score in dup_pairs:
        dup_map.setdefault(a, []).append((os.path.basename(b), score))
        dup_map.setdefault(b, []).append((os.path.basename(a), score))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name","label","stage","reason","valid_traces",
            "dark_ratio","edge_density","green_ratio",
            "duplicate_partner","duplicate_ssim"
        ])
        for fp, out in sorted(results.items(), key=lambda x: os.path.basename(x[0]).lower()):
            reason = out.get("reason", {}) if out else {}
            stage  = reason.get("stage","")
            why    = reason.get("reason","")
            vt     = reason.get("valid_traces","")
            dr     = reason.get("dark_ratio","")
            ed     = reason.get("edge_density","")
            gr     = reason.get("green_ratio","")
            dps    = dup_map.get(fp, [])
            if not dps:
                writer.writerow([os.path.basename(fp), out.get("label",""), stage, why, vt, dr, ed, gr, "", ""])
            else:
                for partner, score in dps:
                    writer.writerow([os.path.basename(fp), out.get("label",""), stage, why, vt, dr, ed, gr, partner, f"{score:.3f}"])

# =========================
# Folder orchestration (NEW)
# =========================
SKIP_DIRS = {"GOOD", "BAD_DUP", "Debug", "debug", "Debug_Overlays"}

def list_pngs_in_dir(dir_path):
    # PNGs directly in dir_path (not recursive)
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(".png")]
    # Sort by numeric index then name
    files = sorted(files, key=lambda p: (extract_numeric_index(os.path.basename(p)) or 10**9, p.lower()))
    return files

def move_file_safe(src, dst_dir):
    ensure_dir(dst_dir)
    base = os.path.basename(src)
    dst  = os.path.join(dst_dir, base)
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    # Avoid overwrite
    if os.path.exists(dst):
        stem, ext = os.path.splitext(base)
        k = 2
        while True:
            cand = os.path.join(dst_dir, f"{stem}_{k}{ext}")
            if not os.path.exists(cand):
                dst = cand
                break
            k += 1
    try:
        shutil.move(src, dst)
    except Exception as e:
        print(f"[WARN] Could not move {src} -> {dst}: {e}")
    return dst

def organize_outputs(dir_path, results, dup_pairs):
    # Build dup set
    dup_set = set()
    for a, b, _ in dup_pairs:
        dup_set.add(a); dup_set.add(b)

    good_dir   = os.path.join(dir_path, "GOOD")
    baddup_dir = os.path.join(dir_path, "BAD_DUP")
    ensure_dir(good_dir); ensure_dir(baddup_dir)

    for fp, out in results.items():
        if not os.path.isfile(fp):
            continue  # Might have been moved already (rare)
        label = out.get("label","")
        if fp in dup_set:
            move_file_safe(fp, baddup_dir)
        else:
            if label == "GOOD":
                move_file_safe(fp, good_dir)
            else:
                move_file_safe(fp, baddup_dir)

def process_one_folder(dir_path):
    print(f"\n=== Processing folder: {dir_path} ===")
    files = list_pngs_in_dir(dir_path)
    if not files:
        print("[Info] No PNGs directly in this folder—skipping.")
        return

    # Per-folder debug dir
    debug_dir = os.path.join(dir_path, "Debug")
    ensure_dir(debug_dir)

    # 1) Classify
    results = {}
    for i, f in enumerate(files, 1):
        out = process_file(f, debug_dir=debug_dir)
        results[f] = out
        label, stage = out["label"], out["reason"].get("stage")
        print(f"[{i}/{len(files)}] {os.path.basename(f)} -> {label} ({stage})")

    # 2) Early summary (txt only, title tag)
    try:
        write_summary_txt(os.path.join(dir_path, "summary.txt"), results, [], title="CLASSIFICATION ONLY")
    except Exception as e:
        print(f"[Summary] Could not write early summary: {e}")

    # 3) Adjacent-only duplicates (UNCHANGED)
    dup_pairs = []
    for idx in range(len(files)-1):
        a, b = files[idx], files[idx+1]
        ra, rb = results[a], results[b]
        ga, gb = ra.get("graph"), rb.get("graph")
        if ga is None or gb is None:
            continue
        if axis_band_equal(ga, gb):
            score = ssim_graph_above_axis(ga, gb)
            if score >= SSIM_THRESHOLD_DUP:
                dup_pairs.append((a, b, score))

    # 4) Final summary (txt + csv)
    write_summary_txt(os.path.join(dir_path, "summary.txt"), results, dup_pairs, title="FINAL")
    write_summary_csv(os.path.join(dir_path, "summary.csv"), results, dup_pairs)
    print(f"[Summary] Written to: {os.path.join(dir_path, 'summary.txt')}")
    print(f"[Summary] Written to: {os.path.join(dir_path, 'summary.csv')}")

    # 5) Auto-organize images
    organize_outputs(dir_path, results, dup_pairs)
    print(f"[Organize] GOOD/ and BAD_DUP/ created in: {dir_path}")

def walk_and_process(root_dir):
    """
    Recursively walk root_dir and process every directory that has PNGs directly inside it.
    Skip internal output directories like GOOD, BAD_DUP, Debug, etc.
    Also process root_dir itself if it contains PNGs.
    """
    # Process root if it has PNGs
    if list_pngs_in_dir(root_dir):
        process_one_folder(root_dir)

    for cur_dir, subdirs, files in os.walk(root_dir):
        # Skip if it's the root (already handled), or internal output dirs
        base = os.path.basename(cur_dir)
        if base in SKIP_DIRS:
            continue
        if cur_dir == root_dir:
            # We'll still iterate subdirs below; do not skip here.
            pass
        # Do not descend into our own output directories later
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]

        # If current dir is the root, we skip directly—it was handled.
        if cur_dir == root_dir:
            continue

        # Process this dir if it has PNGs directly inside it
        if list_pngs_in_dir(cur_dir):
            process_one_folder(cur_dir)

# =========================
# Main orchestrator (GUI picker)
# =========================
def main():
    # GUI folder picker
    root = tk.Tk()
    root.withdraw()
    root.update()
    selected_dir = filedialog.askdirectory(title="Select top-level folder that contains PNGs or subfolders with PNGs")
    root.destroy()

    if not selected_dir:
        print("No folder selected. Exiting.")
        return
    if not os.path.isdir(selected_dir):
        print("Selected path is not a directory. Exiting.")
        return

    walk_and_process(selected_dir)
    print("\nDone. All folders with PNGs have been processed.")

if __name__ == "__main__":
    main()
