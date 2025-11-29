import cv2
import numpy as np
import os
import argparse
import csv
import math
from datetime import timedelta
import json

"""
automatic_croping.py

Usage:
    python automatic_croping.py --video path/to/video.mp4 --timestamps path/to/timestamps.txt --outdir output_frames

What it does:
- Reads a video and a timestamp metadata file.
- For each timestamp, samples a small temporal window of frames around that time to build a robust mask of the visible anatomy.
- Automatically crops away black circular borders and other dark irrelevant borders (and is robust to some occlusions/glare).
- Saves the original frame and the cropped result for each timestamp into the output directory.

Timestamp file formats supported (one timestamp per line):
- seconds as float or int, e.g.:
    12.5
    120
- hh:mm:ss[.ms] e.g.:
    00:01:12.500
- frame:<n> or f<n> or just integer to be interpreted as frame index (if --interpret-ints-as-frames set)
- CSV with first column being a time string is fine

Notes:
- This is a heuristic approach (thresholding + morphology + largest-contour). For very challenging cases
  (heavy occlusion, extreme glare) more advanced ML-based segmentation would be required.
"""


def parse_time_token(tok, interpret_ints_as_frames=False):
    tok = tok.strip()
    if not tok:
        return None
    # frame spec
    if tok.lower().startswith("frame:") or tok.lower().startswith("f:") or tok.lower().startswith("f"):
        # accept "frame:123", "f123", "f:123"
        num = ''.join(ch for ch in tok if ch.isdigit())
        if num:
            return ("frame", int(num))
    # hh:mm:ss(.ms) style
    if ':' in tok:
        parts = tok.split(':')
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h = 0; m, s = parts
            else:
                # fallback
                h = 0; m = 0; s = parts[-1]
            total = h*3600 + m*60 + s
            return ("time", float(total))
        except:
            return None
    # numeric
    try:
        if '.' in tok:
            return ("time", float(tok))
        else:
            i = int(tok)
            if interpret_ints_as_frames:
                return ("frame", i)
            else:
                return ("time", float(i))
    except:
        return None

def read_timestamps_file(path, interpret_ints_as_frames=False):
    results = []
    with open(path, 'r', newline='') as f:
        # try CSV first: read all tokens in first column or all tokens
        try:
            f.seek(0)
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                tok = row[0]
                parsed = parse_time_token(tok, interpret_ints_as_frames)
                if parsed:
                    results.append(parsed)
            if results:
                return results
        except Exception:
            pass
        # fallback read by lines
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_time_token(line, interpret_ints_as_frames)
            if parsed:
                results.append(parsed)
    return results

def secstr(t):
    # format seconds to hh-mm-ss_ms safe filename
    td = timedelta(seconds=float(t))
    s = str(td)
    return s.replace(":", "-").replace(" ", "_").replace(".", "_")

def get_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def time_to_frame_index(time_s, fps):
    return int(round(time_s * fps))

def build_foreground_mask(frames, dark_thresh_v=30, min_saturation=20):
    """
    Given a list of BGR frames (numpy arrays), build a binary mask of non-dark, non-border areas.
    Strategy:
    - Convert to HSV; require Value > dark_thresh_v OR Saturation > min_saturation
    - Combine masks across frames (logical OR) so transient occlusions are reduced
    - Morphological closing to fill small holes, then opening to remove small specks
    """
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)
        # mask non-dark by V and/or sufficiently saturated pixels (to avoid keeping black areas)
        mask_v = (vch > dark_thresh_v).astype(np.uint8)
        mask_s = (sch > min_saturation).astype(np.uint8)
        mask = np.logical_or(mask_v, mask_s).astype(np.uint8) * 255
        combined = np.maximum(combined, mask)
    # morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    # remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        return opened
    # keep largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fg_mask = (labels == largest_label).astype(np.uint8) * 255
    return fg_mask

def find_crop_from_mask(mask, margin=10, square=False):
    # find bounding rect of mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None  # nothing found
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if square:
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        if w > h:
            diff = w - h
            y0 = max(0, y0 - diff // 2)
            y1 = y1 + (diff - diff // 2)
        else:
            diff = h - w
            x0 = max(0, x0 - diff // 2)
            x1 = x1 + (diff - diff // 2)
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = x1 + margin
    y1 = y1 + margin
    return (x0, y0, x1, y1)

def safe_crop(frame, rect):
    if rect is None:
        return None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, min(w-1, x0))
    x1 = max(0, min(w-1, x1))
    y0 = max(0, min(h-1, y0))
    y1 = max(0, min(h-1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1].copy()

def process_video(video_path, timestamps_path, out_dir,
                  window_frames=3, window_seconds=None, interpret_ints_as_frames=False,
                  force_square=False):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    parsed = read_timestamps_file(timestamps_path, interpret_ints_as_frames=interpret_ints_as_frames)
    results_meta = []
    for i, (typ, val) in enumerate(parsed):
        if typ == "frame":
            frame_idx = int(val)
        else:
            frame_idx = time_to_frame_index(float(val), fps)
        # sample a small window of frames around frame_idx
        frames_sample = []
        if window_seconds is not None:
            # select frames within +/- window_seconds around the timestamp
            frame_radius = int(round(window_seconds * fps))
            start_k = frame_idx - frame_radius
            end_k = frame_idx + frame_radius
        else:
            # legacy behavior: use window_frames (odd) centered on frame_idx
            half = window_frames // 2
            start_k = frame_idx - half
            end_k = frame_idx + half
        for k in range(start_k, end_k + 1):
            if k < 0 or k >= total_frames:
                continue
            f = get_frame_at(cap, k)
            if f is not None:
                frames_sample.append(f)
        if not frames_sample:
            print("Warning: no frames for timestamp index {} / {}".format(i, (typ, val)))
            continue
        # build mask and find crop
        mask = build_foreground_mask(frames_sample)
        rect = find_crop_from_mask(mask, margin=10, square=force_square)
        # get reference frame (middle if possible)
        ref_frame = frames_sample[len(frames_sample)//2]
        crop = safe_crop(ref_frame, rect)
        # determine timestamp string for filename
        if typ == "time":
            tstr = secstr(val)
        else:
            tstr = "frame_{}".format(val)
        orig_name = os.path.join(out_dir, f"{i:03d}_{tstr}_orig.jpg")
        cv2.imwrite(orig_name, ref_frame)
        crop_name = None
        if crop is not None:
            crop_name = os.path.join(out_dir, f"{i:03d}_{tstr}_crop.jpg")
            cv2.imwrite(crop_name, crop)
        else:
            print(f"Info: no crop for entry {i} ({typ}:{val}), saved original only")
        results_meta.append({
            "index": i,
            "input": {"type": typ, "value": val},
            "frame_index": frame_idx,
            "orig": orig_name,
            "crop": crop_name,
            "crop_rect": rect
        })
    # save metadata json
    meta_path = os.path.join(out_dir, "crop_results.json")
    with open(meta_path, "w", encoding="utf-8") as jf:
        json.dump(results_meta, jf, indent=2)
    cap.release()
    print("Done. Results in:", out_dir, "Metadata:", meta_path)

def main():
    parser = argparse.ArgumentParser(description="Automatic cropping around visible anatomy using timestamps.")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--timestamps", "-t", required=True, help="Path to timestamp metadata file (one per line)")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for images")
    parser.add_argument("--window-frames", type=int, default=3, help="Temporal window (odd) of frames to sample per timestamp")
    parser.add_argument("--window-sec", type=float, default=None, help="Temporal window in seconds before/after timestamp (overrides --window-frames if set)")
    parser.add_argument("--ints-are-frames", action="store_true", help="Interpret integer tokens as frame indices rather than seconds")
    parser.add_argument("--square", action="store_true", help="Force square crop")
    args = parser.parse_args()
    process_video(args.video, args.timestamps, args.outdir,
                  window_frames=max(1, args.window_frames), window_seconds=args.window_sec,
                  interpret_ints_as_frames=args.ints_are_frames,
                  force_square=args.square)

if __name__ == "__main__":
    main()