"""
main.py

Orchestrate the pipeline:
  1) automatic cropping around timestamps (automatic_croping.py)
  2) quality control of produced images (quality_control_of_images.py)
  3) model inference on surviving images (model.py)

This script invokes the three tools as subprocesses using the same Python interpreter so PATH/import issues
are avoided. It writes a short JSON summary describing the artifact locations which your app can read.

Example (Windows / powershell/cmd):
  python code\main.py --video path\to\video.mp4 --timestamps path\to\timestamps.txt --out-root outputs\run1 \
       --window-sec 1 --checkpoint code\Endo-FM\checkpoints\endofm.pth --device cpu

Notes:
 - This does not attempt to import model internals directly; it calls the existing scripts to avoid PYTHONPATH
   and package import issues with Endo-FM (which expects its own working directory and config parsing).
 - The summary JSON contains the paths to: crop results JSON, QC report, model output JSON, and the image dir.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime


def run_cmd(args, cwd=None):
    print("Running:", " ".join(args))
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)} (code {proc.returncode})")


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline: crop -> QC -> model inference")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--timestamps", required=True, help="Timestamps file (one per line)")
    parser.add_argument("--out-root", default="outputs/run_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help="Root output directory for this run")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Seconds before/after timestamp to select frames")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth) for inference")
    parser.add_argument("--device", default="cpu", help="Device string to pass to model (cpu or cuda:0)")
    parser.add_argument("--move-rejected", action='store_true', help="Move rejected QC frames into a subfolder")
    parser.add_argument("--use-head", action='store_true', help="Run classifier head (logits/probs) instead of features")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    frames_dir = os.path.join(out_root, "frames")
    ensure_dir(frames_dir)

    # 1) Run automatic cropping
    python = sys.executable
    crop_script = os.path.join(os.path.dirname(__file__), "automatic_croping.py")
    crop_cmd = [python, crop_script, "--video", args.video, "--timestamps", args.timestamps,
                "--outdir", frames_dir, "--window-sec", str(args.window_sec)]
    run_cmd(crop_cmd)

    # crop_results.json should be in frames_dir
    crop_meta = os.path.join(frames_dir, "crop_results.json")

    # 2) Run quality control
    qc_script = os.path.join(os.path.dirname(__file__), "quality_control_of_images.py")
    qc_report = os.path.join(out_root, "qc_report.json")
    qc_cmd = [python, qc_script, "--indir", frames_dir, "--report-out", qc_report]
    if args.move_rejected:
        rejected_dir = os.path.join(out_root, "rejected")
        ensure_dir(rejected_dir)
        qc_cmd += ["--move-rejected", rejected_dir]
    run_cmd(qc_cmd)

    # 3) Run model inference on surviving frames (model.py)
    model_script = os.path.join(os.path.dirname(__file__), "model.py")
    model_out = os.path.join(out_root, "model_output.json")
    model_cmd = [python, model_script, "--indir", frames_dir, "--checkpoint", args.checkpoint,
                 "--device", args.device, "--out", model_out]
    if args.use_head:
        model_cmd.append("--use-head")
    run_cmd(model_cmd)

    # Final summary that the app can read
    summary = {
        "frames_dir": frames_dir,
        "crop_meta": crop_meta,
        "qc_report": qc_report,
        "model_output": model_out,
        "rejected_dir": os.path.join(out_root, "rejected") if args.move_rejected else None
    }
    summary_path = os.path.join(out_root, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    print("Pipeline finished. Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == '__main__':
    main()
