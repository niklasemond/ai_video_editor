"""Build a bounded generation job from local inputs and reviewed full-frame masks.

The source frame indices are selected once and retained for compositing, avoiding
independent seek/fps operations choosing different frames. No model is loaded here.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import subprocess


def select_frames(timestamps, start, count, fps, duration, *, pad_last=False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid source duration')
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise ValueError('Missing/nonfinite source timestamps')
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Source timestamps must increase strictly')
    if not math.isfinite(start) or not 0 <= start < duration or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid segment start or frame rate')
    if type(count) is not int or count < 1:
        raise ValueError('Invalid frame count')
    targets = [start + i/fps for i in range(count)]
    if type(pad_last) is not bool:
        raise ValueError('pad_last must be boolean')
    if not pad_last and targets[-1] >= duration:
        raise ValueError('Requested frame times extend outside the source')
    if pad_last and sum(t > timestamps[-1] + 1e-6 for t in targets) > 3:
        raise ValueError('Only up to three final padding frames are permitted')
    selected = []
    for target in targets:
        right = bisect.bisect_left(timestamps, target)
        choices = [i for i in (right-1, right) if 0 <= i < len(timestamps)]
        selected.append(min(choices, key=lambda i: (abs(timestamps[i]-target), i)))
    duplicates = [b for a,b in zip(selected,selected[1:]) if a == b]
    if duplicates and (not pad_last or len(duplicates) > 3 or any(i != len(timestamps)-1 for i in duplicates)):
        raise ValueError('Requested frame rate duplicates source frames; lower it')
    return selected


def validate_box(box, size):
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('Crop must contain four integer coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError('Crop extends outside the input')


def validate_mask_crop(mask, crop, dilation=0):
    box = mask.convert('L').getbbox()
    if box is None:
        return
    x0, y0, x1, y1 = box
    expanded = (max(0,x0-dilation), max(0,y0-dilation),
                min(mask.width,x1+dilation), min(mask.height,y1+dilation))
    if not (crop[0] <= expanded[0] and crop[1] <= expanded[1]
            and crop[2] >= expanded[2] and crop[3] >= expanded[3]):
        raise ValueError('Performer mask extends outside the generation crop; widen it or correct the mask')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def continuation_inputs(previous, config, reference, count):
    from PIL import Image
    if config.get('engine') != 'animate' or count not in (1, 5):
        raise ValueError('Native continuation requires Animate and one or five frames')
    if config['frames'] <= count:
        raise ValueError('Continuation leaves no new frames to generate')
    if (previous/'INCOMPLETE').exists():
        raise ValueError('Previous job is incomplete')
    old = json.loads((previous/'config.json').read_text())
    for key in ('source_sha256', 'engine', 'model', 'fps', 'width', 'height', 'crop'):
        if old.get(key) != config.get(key):
            raise ValueError('Continuation source or crop configuration changed: '+key)
    if len(old.get('source_frame_indices', [])) < count or config['source_frame_indices'][:count] != old['source_frame_indices'][-count:]:
        raise ValueError('Continuation frames do not align with source frame indices')
    with Image.open(previous/'reference.png') as old_reference:
        if old_reference.size != reference.size or old_reference.convert('RGB').tobytes() != reference.tobytes():
            raise ValueError('Continuation reference image changed')
    generated = sorted((previous/'generated').glob('*.png'))
    if len(generated) != old['frames']:
        raise ValueError('Previous generation is incomplete')
    for path in generated[-count:]:
        with Image.open(path) as frame:
            frame.load()
            if frame.size != reference.size:
                raise ValueError('Continuation frame dimensions disagree')
    return generated[-count:]


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'reference', 'masks', 'settings', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--composite-masks', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--continue-frames', type=int, choices=(1, 5), default=5)
    args = parser.parse_args()
    from PIL import Image, ImageOps, ImageFilter
    from local_paths import validate_local_output
    from validation import validate_job
    validate_local_output(args.output)
    if args.output.exists():
        raise ValueError('Output already exists; choose a new job directory')
    if not args.source.is_file() or not args.reference.is_file():
        raise ValueError('Missing source video or reference image')
    config = json.loads(args.settings.read_text())
    if type(config.get('frames')) is not int or not 5 <= config['frames'] <= 25:
        raise ValueError('Prepare bounded jobs of 5–25 frames')
    if (config['frames'] - 1) % 4:
        raise ValueError('Frame count must be 4n+1')
    size = (config['width'], config['height'])
    if any(type(v) is not int or not 16 <= v <= 1024 or v % 16 for v in size):
        raise ValueError('Generation dimensions must be 16-multiples up to 1024')
    dilate = config.get('mask_dilate', 0)
    if type(dilate) is not int or not 0 <= dilate <= 16:
        raise ValueError('Mask dilation must be 0–16 source pixels')
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams',
        'v:0', '-show_entries', 'stream=width,height,duration:frame=best_effort_timestamp_time',
        '-of', 'json', str(args.source)], text=True))
    stream = info['streams'][0]
    source_size = (stream['width'], stream['height'])
    timestamps = [float(f['best_effort_timestamp_time']) for f in info['frames']]
    indices = select_frames(timestamps, config['start'], config['frames'],
                            config['fps'], float(stream['duration']),
                            pad_last=config.get('pad_last', False))
    validate_box(config['crop'], source_size)
    reference = ImageOps.exif_transpose(Image.open(args.reference)).convert('RGB')
    if 'reference_crop' in config:
        validate_box(config['reference_crop'], reference.size)
        reference = reference.crop(config['reference_crop'])
    reference = ImageOps.pad(reference, size, method=Image.Resampling.LANCZOS, color=(0,0,0))
    config.update(source_prepared=True, source_sha256=digest(args.source),
        source_frame_indices=indices, source_frame_pts=[timestamps[i] for i in indices],
        resize_to_crop=True, composite_masks=bool(args.composite_masks),
        padded_tail_frames=len(indices)-len(set(indices)),
        output_duration=min(config['frames']/config['fps'],float(stream['duration'])-config['start']),
        continue_motion_frames=args.continue_frames if args.continue_from else 0)
    continuation = continuation_inputs(args.continue_from, config, reference,
        args.continue_frames) if args.continue_from else []
    mask_folders = [(args.masks, 'masks')]
    if args.composite_masks:
        mask_folders.append((args.composite_masks, 'composite-masks'))
    for folder, name in mask_folders:
        if (folder/'INCOMPLETE').exists():
            raise ValueError('Refusing incomplete tracking results')
        for index in indices:
            with Image.open(folder/f'{index:05}.png') as mask:
                mask.load()
                if mask.size != source_size:
                    raise ValueError('Mask and source video dimensions disagree')
                validate_mask_crop(mask,config['crop'],dilate if name == 'masks' else 0)
    args.output.mkdir(parents=True)
    (args.output/'INCOMPLETE').write_text('Job preparation is incomplete.\n')
    for folder in ['source-frames', 'frames'] + [name for _, name in mask_folders]:
        (args.output/folder).mkdir()
    unique_indices = list(dict.fromkeys(indices))
    expression = '+'.join(f'eq(n\\,{i})' for i in unique_indices)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(args.source),
        '-vf', 'select='+expression, '-fps_mode', 'passthrough', '-frames:v', str(len(unique_indices)),
        '-start_number', '0', str(args.output/'source-frames/%03d.png')], check=True)
    sources = sorted((args.output/'source-frames').glob('*.png'))
    if len(sources) != len(unique_indices):
        raise RuntimeError('Source decoder did not return all selected frames')
    for i in range(len(unique_indices), len(indices)):
        path = args.output/'source-frames'/f'{i:03}.png'
        path.write_bytes(sources[-1].read_bytes())
        sources.append(path)
    for n, (source, index) in enumerate(zip(sources, indices)):
        Image.open(source).convert('RGB').crop(config['crop']).resize(size,
            Image.Resampling.LANCZOS).save(args.output/'frames'/f'{n:03}.png')
        for folder, name in mask_folders:
            mask = Image.open(folder/f'{index:05}.png').convert('L')
            if dilate and name == 'masks':
                mask = mask.filter(ImageFilter.MaxFilter(2*dilate+1))
            mask.crop(config['crop']).resize(size, Image.Resampling.NEAREST).save(
                args.output/name/f'{n:03}.png')
    reference.save(args.output/'reference.png')
    if continuation:
        (args.output/'continuation').mkdir()
        for i, frame in enumerate(continuation):
            (args.output/'continuation'/f'{i:03}.png').write_bytes(frame.read_bytes())
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    validate_job(args.output, config, preparing=True)
    (args.output/'INCOMPLETE').unlink()
    print('Prepared', len(indices), 'source-aligned frames; masks still require visual review')


if __name__ == '__main__':
    main()
