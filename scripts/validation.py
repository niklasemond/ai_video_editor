"""Fail before model loading on invalid experimental job inputs."""
from pathlib import Path
from PIL import Image


def validate_job(job, config):
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
