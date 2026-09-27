"""Offline CPU SAM2 tracking from reviewed point prompts, with separate output.

Run under resource_watch.py and the same process-local network-denial profile as
generation. Output masks require visual review before conditioning/compositing.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def validate_prompts(prompts, count, size):
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('At least one reviewed point prompt is required')
    seen = set()
    for item in prompts:
        frame = item.get('frame')
        points, labels = item.get('points', []), item.get('labels', [])
        if type(frame) is not int or not 0 <= frame < count or frame in seen:
            raise ValueError('Prompt frame is invalid or repeated')
        seen.add(frame)
        if not points or len(points) != len(labels) or 1 not in labels:
            raise ValueError('Each prompt needs matching points/labels and a positive point')
        for point, label in zip(points, labels):
            if label not in (0, 1) or len(point) != 2:
                raise ValueError('Invalid point or label')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ValueError('Coordinates must be finite numbers')
            if not (0 <= point[0] < size[0] and 0 <= point[1] < size[1]):
                raise ValueError('Point outside image')
    if 0 not in seen:
        raise ValueError('A first-frame prompt is required for forward propagation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames', type=Path)
    parser.add_argument('prompts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from local_paths import validate_local_output
    from download_models import verify
    from PIL import Image
    validate_local_output(args.output)
    if args.output.exists():
        raise ValueError('Output already exists; use a new directory to preserve prior results')
    frames = sorted(args.frames.glob('*.jpg'))
    if not frames or [int(p.stem) for p in frames] != list(range(len(frames))):
        raise ValueError('Frames must be contiguous numeric JPEGs starting at zero')
    size = Image.open(frames[0]).size
    for path in frames:
        with Image.open(path) as frame:
            frame.load()
            if frame.size != size:
                raise ValueError('Frame dimensions must be consistent')
    prompts = json.loads(args.prompts.read_text())
    validate_prompts(prompts, len(frames), size)
    item = next(x for x in json.loads((ROOT/'models.lock.json').read_text())
                if x['filename'] == 'sam2.1_hiera_tiny.pt')
    checkpoint = ROOT/'.local/ComfyUI/models'/item['destination']
    if not verify(checkpoint, item):
        raise RuntimeError('Missing/corrupt SAM2 checkpoint')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TORCH_HOME=str(ROOT/'.cache/torch'))
    sys.path.insert(0, str(ROOT/'.local/SAM2'))
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4)
    start = time.monotonic()
    # No CUDA extension or compiled predictor; native CPU FP32 operations.
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
        str(checkpoint), device='cpu', apply_postprocessing=False)
    args.output.mkdir(parents=True)
    (args.output/'INCOMPLETE').write_text('Do not use until complete and visually reviewed.\n')
    stats = []
    with torch.inference_mode():
        state = predictor.init_state(str(args.frames), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        for item in sorted(prompts, key=lambda x: x['frame']):
            predictor.add_new_points_or_box(state, item['frame'], 1,
                points=np.array(item['points'], dtype=np.float32),
                labels=np.array(item['labels'], dtype=np.int32))
        for index, ids, logits in predictor.propagate_in_video(state):
            if ids != [1] or not torch.isfinite(logits).all():
                raise RuntimeError('Unexpected object IDs or nonfinite mask')
            mask = (logits[0, 0] > 0).cpu().numpy()
            pixels = int(mask.sum())
            image = Image.fromarray(mask.astype(np.uint8)*255)
            part = args.output/f'{index:05}.tmp.png'
            image.save(part)
            part.replace(args.output/f'{index:05}.png')
            stats.append({'frame': index, 'pixels': pixels, 'bbox': image.getbbox()})
            print('Mask', index, 'pixels', pixels, flush=True)
    if len(stats) != len(frames):
        raise RuntimeError('Incomplete propagation')
    (args.output/'stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    (args.output/'INCOMPLETE').unlink()
    print('SEGMENT COMPLETE', round(time.monotonic()-start, 2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
