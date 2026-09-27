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


def select_frames(timestamps, start, count, fps, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid source duration')
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise ValueError('Missing/nonfinite source timestamps')
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Source timestamps must increase strictly')
    if not math.isfinite(start) or start < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid segment start or frame rate')
    if type(count) is not int or count < 1 or start + (count-1)/fps >= duration:
        raise ValueError('Requested frame times extend outside the source')
    selected = []
    for target in (start + i/fps for i in range(count)):
        right = bisect.bisect_left(timestamps, target)
        choices = [i for i in (right-1, right) if 0 <= i < len(timestamps)]
        selected.append(min(choices, key=lambda i: (abs(timestamps[i]-target), i)))
    if len(set(selected)) != count:
        raise ValueError('Requested frame rate duplicates source frames; lower it')
    return selected


def validate_box(box, size):
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('Crop must contain four integer coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError('Crop extends outside the input')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'reference', 'masks', 'settings', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--composite-masks', type=Path)
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
                            config['fps'], float(stream['duration']))
    validate_box(config['crop'], source_size)
    reference = ImageOps.exif_transpose(Image.open(args.reference)).convert('RGB')
    if 'reference_crop' in config:
        validate_box(config['reference_crop'], reference.size)
        reference = reference.crop(config['reference_crop'])
    mask_folders = [(args.masks, 'masks')]
    if args.composite_masks:
        mask_folders.append((args.composite_masks, 'composite-masks'))
    for folder, _ in mask_folders:
        if (folder/'INCOMPLETE').exists():
            raise ValueError('Refusing incomplete tracking results')
        for index in indices:
            with Image.open(folder/f'{index:05}.png') as mask:
                mask.load()
                if mask.size != source_size:
                    raise ValueError('Mask and source video dimensions disagree')
    args.output.mkdir(parents=True)
    (args.output/'INCOMPLETE').write_text('Job preparation is incomplete.\n')
    for folder in ['source-frames', 'frames'] + [name for _, name in mask_folders]:
        (args.output/folder).mkdir()
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(args.source),
        '-vf', 'select='+expression, '-fps_mode', 'passthrough', '-frames:v', str(len(indices)),
        '-start_number', '0', str(args.output/'source-frames/%03d.png')], check=True)
    sources = sorted((args.output/'source-frames').glob('*.png'))
    if len(sources) != len(indices):
        raise RuntimeError('Source decoder did not return all selected frames')
    for n, (source, index) in enumerate(zip(sources, indices)):
        Image.open(source).convert('RGB').crop(config['crop']).resize(size,
            Image.Resampling.LANCZOS).save(args.output/'frames'/f'{n:03}.png')
        for folder, name in mask_folders:
            mask = Image.open(folder/f'{index:05}.png').convert('L')
            if dilate and name == 'masks':
                mask = mask.filter(ImageFilter.MaxFilter(2*dilate+1))
            mask.crop(config['crop']).resize(size, Image.Resampling.NEAREST).save(
                args.output/name/f'{n:03}.png')
    ImageOps.pad(reference, size, method=Image.Resampling.LANCZOS, color=(0,0,0)).save(
        args.output/'reference.png')
    config.update(source_prepared=True, source_sha256=digest(args.source),
        source_frame_indices=indices, source_frame_pts=[timestamps[i] for i in indices],
        resize_to_crop=True, composite_masks=bool(args.composite_masks))
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    validate_job(args.output, config, preparing=True)
    (args.output/'INCOMPLETE').unlink()
    print('Prepared', len(indices), 'source-aligned frames; masks still require visual review')


if __name__ == '__main__':
    main()
