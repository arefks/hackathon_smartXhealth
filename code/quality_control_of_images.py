"""
quality_control_of_images.py

Group-based quality filtering for batches of frames produced by automatic_croping.py.

Usage examples:
    python quality_control_of_images.py --indir path/to/output_frames --report-out qc_report.json --move-rejected rejected/

Behavior:
- Groups images by the numeric index prefix produced by `automatic_croping.py` (e.g. "000_*", "001_*").
- For each group, computes simple image-quality metrics:
    - blur (Laplacian variance)
    - noise estimate (std of high-frequency residual)
    - empty/dark frame detection (low mean / low variance / low saturation)
- Marks frames failing thresholds as rejected and optionally moves them to a `--move-rejected` directory while producing a JSON report.

Notes:
- Thresholds are conservative defaults; tune them for your data.
- This is a lightweight heuristic filter designed to be fast and robust; for production you may want an ML-based classifier.
"""

import os
import cv2
import numpy as np
import argparse
import json
import shutil
import re
from collections import defaultdict


def variance_of_laplacian(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def estimate_noise(image_gray):
    """Estimate noise by subtracting a Gaussian-smoothed version and computing std dev of residual."""
    blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
    residual = image_gray.astype(np.float32) - blur.astype(np.float32)
    return residual.std()


def fraction_dark_pixels(gray_image, thresh=10):
    return np.mean(gray_image <= thresh)


def fraction_low_saturation(bgr_image, sat_thresh=20):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    return np.mean(s <= sat_thresh)


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = variance_of_laplacian(gray)
    noise_score = estimate_noise(gray)
    dark_frac = fraction_dark_pixels(gray)
    low_sat_frac = fraction_low_saturation(img)
    mean_intensity = float(gray.mean())
    var_intensity = float(gray.var())
    return {
        "path": path,
        "blur_var": float(blur_score),
        "noise_std": float(noise_score),
        "dark_fraction": float(dark_frac),
        "low_saturation_fraction": float(low_sat_frac),
        "mean_intensity": mean_intensity,
        "var_intensity": var_intensity,
    }


def filter_group(image_paths, blur_thresh=100.0, noise_thresh=12.0, dark_frac_thresh=0.9, low_sat_thresh=0.95,
                 min_variance=50.0):
    """
    Given a list of image file paths belonging to one timestamp group, analyze and return accepted and rejected lists.
    Thresholds (defaults) are conservative; tune for your data.
    """
    results = []
    for p in image_paths:
        stats = analyze_image(p)
        if stats is None:
            results.append((p, False, {"reason": "cannot_read"}))
            continue
        reject_reasons = []
        if stats["blur_var"] < blur_thresh:
            reject_reasons.append("blurry")
        if stats["noise_std"] > noise_thresh:
            reject_reasons.append("noisy")
        if stats["dark_fraction"] > dark_frac_thresh:
            reject_reasons.append("dark")
        if stats["low_saturation_fraction"] > low_sat_thresh and stats["mean_intensity"] < 40:
            # very low saturation combined with low intensity likely empty/occluded
            reject_reasons.append("low_saturation_or_empty")
        if stats["var_intensity"] < min_variance:
            # nearly uniform frame
            reject_reasons.append("low_variance")
        ok = (len(reject_reasons) == 0)
        results.append((p, ok, {"reasons": reject_reasons, "stats": stats}))
    accepted = [r for r in results if r[1]]
    rejected = [r for r in results if not r[1]]
    return accepted, rejected


def group_images_by_prefix(indir, pattern=r"^(\d{1,6})_"):
    files = [f for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f))]
    groups = defaultdict(list)
    prog = re.compile(pattern)
    for fn in files:
        m = prog.match(fn)
        if m:
            gid = m.group(1)
        else:
            # put into special group "misc"
            gid = "misc"
        groups[gid].append(os.path.join(indir, fn))
    # sort paths for determinism
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])
    return groups


def process_directory(indir, report_out=None, move_rejected=None,
                      blur_thresh=100.0, noise_thresh=12.0, dark_frac_thresh=0.9,
                      low_sat_thresh=0.95, min_variance=50.0):
    groups = group_images_by_prefix(indir)
    report = {"indir": indir, "groups": {}}
    for gid, paths in sorted(groups.items(), key=lambda x: x[0]):
        accepted, rejected = filter_group(paths, blur_thresh=blur_thresh, noise_thresh=noise_thresh,
                                          dark_frac_thresh=dark_frac_thresh, low_sat_thresh=low_sat_thresh,
                                          min_variance=min_variance)
        report["groups"][gid] = {
            "total": len(paths),
            "accepted": [r[0] for r in accepted],
            "rejected": [{"path": r[0], "reasons": r[2].get("reasons", []), "stats": r[2].get("stats")} for r in rejected]
        }
        if move_rejected and rejected:
            os.makedirs(move_rejected, exist_ok=True)
            for r in rejected:
                src = r[0]
                dst = os.path.join(move_rejected, os.path.basename(src))
                try:
                    shutil.move(src, dst)
                except Exception:
                    # fallback to copy
                    try:
                        shutil.copy2(src, dst)
                        os.remove(src)
                    except Exception:
                        pass
    if report_out:
        with open(report_out, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description="Quality-control image batches produced by automatic_croping.py")
    parser.add_argument("--indir", required=True, help="Directory with images (automatic_croping output)")
    parser.add_argument("--report-out", help="Path to JSON report (saved if provided)")
    parser.add_argument("--move-rejected", help="Directory to move rejected frames into")
    parser.add_argument("--blur-thresh", type=float, default=100.0)
    parser.add_argument("--noise-thresh", type=float, default=12.0)
    parser.add_argument("--dark-frac-thresh", type=float, default=0.9)
    parser.add_argument("--low-sat-thresh", type=float, default=0.95)
    parser.add_argument("--min-variance", type=float, default=50.0)
    args = parser.parse_args()
    report = process_directory(args.indir, report_out=args.report_out, move_rejected=args.move_rejected,
                               blur_thresh=args.blur_thresh, noise_thresh=args.noise_thresh,
                               dark_frac_thresh=args.dark_frac_thresh, low_sat_thresh=args.low_sat_thresh,
                               min_variance=args.min_variance)
    print("QC done. Groups:" , len(report.get("groups", {})))


if __name__ == '__main__':
    main()
