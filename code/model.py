"""
model.py

Load a pretrained Endo-FM model and run inference on selected images (single group or multiple groups).

Usage examples:
  # run on a list of images (treated as a temporal sequence)
  python code/model.py --images img1.jpg img2.jpg --checkpoint code/Endo-FM/checkpoints/endofm.pth --device cpu --out results.json

  # run on all image groups in a directory (prefix grouping like automatic_croping)
  python code/model.py --indir code/output_frames --checkpoint code/Endo-FM/checkpoints/endofm.pth --device cuda:0 --out results.json

Outputs a JSON containing per-group feature vectors and optional class scores.
"""

import os
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch

# import project model builders
from models import build_vit_base_patch16_224, get_vit_base_patch16_224
from models.helpers import load_checkpoint
from utils.parser import parse_args, load_config
from models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def group_images_by_prefix(indir, pattern_prefix_digits=6):
    files = [f for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f))]
    groups = defaultdict(list)
    for fn in files:
        # try to extract leading numeric prefix before underscore
        parts = fn.split('_', 1)
        if len(parts) > 1 and parts[0].isdigit():
            gid = parts[0]
        else:
            gid = "misc"
        groups[gid].append(os.path.join(indir, fn))
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def preprocess_images(paths, crop_size=224, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    # returns tensor shape (1, 3, T, H, W)
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h != crop_size or w != crop_size:
            img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        # normalize
        img = (img - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)
        # to C,H,W
        img = img.transpose(2, 0, 1)
        imgs.append(img)
    if len(imgs) == 0:
        return None
    # stack into (C, T, H, W)
    arr = np.stack(imgs, axis=1)
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor


def build_model_and_load(checkpoint_path=None, device='cpu'):
    # build model using repo helper
    # use parser to load default config used by build helper
    opt = parse_args()
    # set config file to TimeSformer default used by repo
    opt.cfg_file = "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    cfg = load_config(opt)

    # build model
    try:
        model = get_vit_base_patch16_224(cfg)
    except Exception:
        # fallback: use build helper if available
        model = build_vit_base_patch16_224()

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # load checkpoint (helpers.load_checkpoint will map to cpu)
        load_checkpoint(model, checkpoint_path)

    model.eval()
    model.to(device)
    return model, cfg


def infer_on_group(model, cfg, image_paths, device='cpu', use_head=False):
    tensor = preprocess_images(image_paths, crop_size=cfg.DATA.TRAIN_CROP_SIZE)
    if tensor is None:
        return None
    tensor = tensor.to(device)
    with torch.no_grad():
        if use_head:
            out = model(tensor, use_head=True)
            # if out is logits, convert to probabilities
            if out.dim() == 2:
                probs = torch.softmax(out, dim=1).cpu().numpy().tolist()
                return {"logits": out.cpu().numpy().tolist(), "probs": probs}
            else:
                return {"output": out.cpu().numpy().tolist()}
        else:
            feats = model(tensor)
            return {"features": feats.cpu().numpy().tolist()}


def main():
    parser = argparse.ArgumentParser(description="Run Endo-FM model on selected images or image groups")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs='+', help="Paths to images to treat as a single temporal sequence")
    group.add_argument("--indir", help="Directory with images (grouped by numeric prefix)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", default='cpu', help="Device to run on, e.g. cpu or cuda:0")
    parser.add_argument("--use-head", action='store_true', help="Run classifier head and return logits/probs instead of features")
    parser.add_argument("--out", default='model_output.json', help="JSON output path")
    args = parser.parse_args()

    model, cfg = build_model_and_load(args.checkpoint, device=args.device)

    results = {}
    if args.images:
        res = infer_on_group(model, cfg, args.images, device=args.device, use_head=args.use_head)
        results['single'] = {'paths': args.images, 'result': res}
    else:
        groups = group_images_by_prefix(args.indir)
        for gid, paths in groups.items():
            res = infer_on_group(model, cfg, paths, device=args.device, use_head=args.use_head)
            results[gid] = {'paths': paths, 'result': res}

    with open(args.out, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=2)
    print(f"Saved output to {args.out}")


if __name__ == '__main__':
    main()
