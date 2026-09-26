"""Extract CPU DWPose controls for a manually selected, masked performer.

Uses pinned upstream preprocessing/drawing and a pinned ONNX model. Never calls
from_pretrained or a download helper. Review the resulting pose/face images.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT / '.cache')
sys.path.insert(0, str(ROOT / '.local/controlnet_aux/src'))
import numpy as np
import onnxruntime as ort
from PIL import Image
from custom_controlnet_aux.dwpose import draw_poses
from custom_controlnet_aux.dwpose.wholebody import Wholebody
from custom_controlnet_aux.dwpose.dw_onnx.cv_ox_pose import inference_pose
from download_models import verify


def main():
    job = Path(sys.argv[1])
    item = next(x for x in json.loads((ROOT/'models.lock.json').read_text())
                if x['filename'] == 'dw-ll_ucoco_384.onnx')
    weight = ROOT / '.local/ComfyUI/models' / item['destination']
    if not verify(weight, item):
        raise RuntimeError('Missing/corrupt pose weight')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(weight), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    for folder in ('poses', 'faces'):
        (job/folder).mkdir(exist_ok=True)
    all_points = []
    for i, path in enumerate(sorted((job/'frames').glob('*.png'))):
        frame = Image.open(path).convert('RGB')
        mask = Image.open(job/'masks'/path.name).convert('L')
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError('Empty performer mask')
        points, scores = inference_pose(session, [bbox], np.asarray(frame)[:,:,::-1].copy())
        info = np.concatenate((points, scores[...,None]), axis=-1)
        neck = np.mean(info[:,[5,6]], axis=1)
        neck[:,2] = np.logical_and(info[:,5,2] > .3, info[:,6,2] > .3)
        info = np.insert(info,17,neck,axis=1)
        info[:,[1,2,3,4,6,7,8,9,10,12,13,14,15,16,17]] = info[:,[17,6,8,10,7,9,12,14,16,13,15,2,1,4,3]]
        all_points.append(info.tolist())
        poses = Wholebody.format_result(info)
        canvas = draw_poses(poses, frame.height, frame.width, draw_face=False)
        Image.fromarray(canvas).save(job/'poses'/f'{i:03}.png')
        face = info[0,24:92]
        face = face[face[:,2] > .3]
        if len(face) < 5:
            raise RuntimeError(f'Insufficient face landmarks in frame {i}')
        lo, hi = face[:,:2].min(0), face[:,:2].max(0)
        center = (lo+hi)/2
        side = max(hi-lo)*1.5
        box = tuple(np.round([center[0]-side/2, center[1]-side/2,
                              center[0]+side/2, center[1]+side/2]).astype(int))
        frame.crop(box).resize((512,512),Image.Resampling.LANCZOS).save(job/'faces'/f'{i:03}.png')
        print('Pose/face', i, 'complete', flush=True)
    (job/'pose-keypoints.json').write_text(json.dumps(all_points))


if __name__ == '__main__':
    main()
