"""Fail before model loading on invalid experimental job inputs."""
from pathlib import Path
from PIL import Image


def validate_vace_keys(keys):
    required = {'vace_patch_embedding.weight', 'vace_patch_embedding.bias',
                'vace_blocks.0.before_proj.weight', 'vace_blocks.0.after_proj.weight'}
    missing = required - set(keys)
    if missing:
        raise ValueError('Incomplete VACE checkpoint: ' + ', '.join(sorted(missing)))


def flatten_video_frames(pixels, expected_shape):
    if pixels.ndim == 5:
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
    if tuple(pixels.shape) != tuple(expected_shape):
        raise ValueError(f'Unexpected decoded dimensions: {tuple(pixels.shape)}')
    return pixels


def validate_motion_controls(job, config):
    for folder, size in [('poses', (config['width'], config['height'])), ('faces', (512, 512))]:
        paths = sorted((Path(job)/folder).glob('*.png'))
        if len(paths) != config['frames']:
            raise ValueError('Incomplete motion controls: ' + folder)
        for path in paths:
            with Image.open(path) as im:
                im.load()
                if im.size != size:
                    raise ValueError('Wrong motion-control dimensions: ' + folder)


def validate_job(job, config):
    if not isinstance(config.get('tiled_vae', False), bool):
        raise ValueError('tiled_vae must be boolean')
    if not isinstance(config.get('composite_masks', False), bool):
        raise ValueError('composite_masks must be boolean')
    if config.get('text_dtype', 'fp16') not in ('fp16', 'fp32'):
        raise ValueError('Unsupported text encoder precision')
    models = {
        'vace': ('Wan2.1-VACE-1.3B-Q8_0.gguf', 'wan2.1_vace_1.3B_fp16.safetensors'),
        'animate': ('Wan2.2-Animate-14B-Q2_K.gguf', 'Wan2.2-Animate-14B-Q3_K_S.gguf'),
    }
    allowed = models.get(config.get('engine', 'vace'))
    if allowed is None or config.get('model', allowed[0]) not in allowed:
        raise ValueError('Unsupported model or engine combination')
    for name in ('width', 'height'):
        value = config[name]
        if not isinstance(value, int) or value < 16 or value % 16:
            raise ValueError(f'{name} must be a positive multiple of 16')
    n = config['frames']
    if not isinstance(n, int) or n < 5 or (n-1) % 4:
        raise ValueError('Frame count must be 4n+1 and at least five')
    if not 1 <= config['steps'] <= 100 or not 0 < config['cfg'] <= 20:
        raise ValueError('Invalid steps or guidance')
    if not config['positive'].strip():
        raise ValueError('A replacement description is required')
    size = (config['width'], config['height'])
    for folder in ('frames', 'masks'):
        paths = sorted((Path(job)/folder).glob('*.png'))
        if len(paths) != n:
            raise ValueError(f'Expected {n} {folder}; got {len(paths)}')
        for p in paths:
            with Image.open(p) as im:
                im.load()
                if im.size != size:
                    raise ValueError(f'Wrong dimensions in {folder}')
    with Image.open(Path(job)/'reference.png') as im:
        im.load()
        if im.size != size:
            raise ValueError('Reference must be padded to generation dimensions')
