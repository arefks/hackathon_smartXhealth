# hackathon_smartXhealth

A small project scaffold and pipeline for working with Endo-FM (foundation model for endoscopy video analysis).

This repository provides a lightweight orchestration around the Endo-FM codebase to:

- extract temporal frames around annotated timestamps (automatic cropping),
- filter out low-quality frames (blurred, noisy, empty), and
- run inference with a pretrained Endo-FM model to produce features or predictions.

The intent is to produce a clean set of artifacts your application can consume (images, QC report, model outputs) and a single JSON summary that points to these artifacts.

## Highlights

- Top-level `code/` contains pipeline scripts: `automatic_croping.py`, `quality_control_of_images.py`, `model.py`, and the orchestration `main.py`.
- The official Endo-FM implementation is included under `code/Endo-FM/` (submodule/clone). It contains model definitions, configs, and utilities.
- A Windows-friendly conda file `code/Endo-FM/environment-windows.yaml` is provided; for training or full reproducibility we recommend WSL2 / a Linux environment.

## Folder layout

- `code/` — pipeline scripts and the Endo-FM code (under `code/Endo-FM/`).
- `inputs/` — place raw uploads or videos here (tracked via `.gitkeep`).
- `outputs/` — pipeline outputs and run artifacts (created per run).
- `docs/` — project documentation and usage notes.

## Quick start (recommended)

1. Prepare environment

	- Best: use WSL2 (Ubuntu) or a Linux server for compatibility with training scripts and bash helpers.
	- For Windows-only/testing, create the provided Windows environment:

	  ```powershell
	  cd code\Endo-FM
	  conda env create -f environment-windows.yaml
	  conda activate endofm-windows
	  ```

2. Place the pretrained checkpoint

	- Download your Endo-FM checkpoint and save it to `code/Endo-FM/checkpoints/`.
	- Example path used by the pipeline: `code/Endo-FM/checkpoints/endofm.pth`.

3. Run the pipeline

	```powershell
	python code\main.py --video inputs\case1.mp4 --timestamps inputs\case1_timestamps.txt \
		 --out-root outputs\run1 --window-sec 1 --checkpoint code\Endo-FM\checkpoints\endofm.pth --device cpu --move-rejected
	```

	The script executes three stages (cropping → QC → model inference) and writes `pipeline_summary.json` into the `--out-root` directory. That summary is the single file your app should read to locate outputs.

## Outputs produced

After a run you will find (example `outputs/run1`):

- `frames/` — extracted original and cropped images per timestamp
- `frames/crop_results.json` — metadata with frame indices, crop rectangles and saved filenames
- `qc_report.json` — per-group QC report listing accepted/rejected frames and reasons
- `model_output.json` — model features or logits for each group
- `rejected/` — (optional) moved rejected images
- `pipeline_summary.json` — single-source-of-truth JSON pointing to the above files

Read `pipeline_summary.json` to locate and serve artifacts to your application.

## Notes & recommendations

- The Endo-FM repository under `code/Endo-FM/` is the original project; use its `environment.yaml` for reproduction on Linux, or the provided `environment-windows.yaml` for Windows convenience.
- The pipeline scripts favor subprocess-based orchestration to avoid import/path surprises with the Endo-FM code. If you'd like tighter integration (in-process inference), we can adapt `model.py` to be importable.
- Thresholds used by QC are conservative defaults — tune them in `code/quality_control_of_images.py` or pass CLI overrides where supported.
- Always verify the checkpoint you downloaded matches the model architecture/config; mismatches may still load but yield unexpected outputs.

## Next steps I can do for you

- Add a small `docs/USAGE-APP.md` describing how an external app should read `pipeline_summary.json` and stream artifacts.
- Convert the pipeline to run in-process (no subprocess calls) for lower latency.
- Add unit tests for the QC heuristics and a small end-to-end smoke test using a short sample video.

If you'd like one of those, tell me which and I'll implement it next.

---

Licensed for internal hackathon use.
