"""Composite a tested crop into source frames and retain synchronized source audio."""
import argparse
import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageFilter
from local_paths import validate_local_output


def composite(source, generated, mask, box, resize_to_crop=False):
    x0,y0,x1,y1 = box
    if mask.size != generated.size:
        raise ValueError('Mask and generated frame dimensions disagree')
    if resize_to_crop:
        generated = generated.resize((x1-x0,y1-y0), Image.Resampling.LANCZOS)
        mask = mask.resize((x1-x0,y1-y0), Image.Resampling.BILINEAR)
    if generated.size != (x1-x0,y1-y0) or mask.size != generated.size:
        raise ValueError('Crop, mask and generated frame dimensions disagree')
    if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
        raise ValueError('Crop outside source')
    result = source.copy()
    region = Image.composite(generated, source.crop(box), mask)
    result.paste(region,(x0,y0))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=pathlib.Path)
    parser.add_argument('source',type=pathlib.Path)
    args=parser.parse_args()
    validate_local_output(args.job)
    if not args.source.is_file():raise ValueError('Source video missing')
    cfg=json.loads((args.job/'config.json').read_text())
    generated=sorted((args.job/'generated').glob('*.png'))
    masks=sorted((args.job/'masks').glob('*.png'))
    if len(generated)!=cfg['frames'] or len(masks)!=cfg['frames']:
        raise ValueError('Incomplete generation or mask sequence')
    original=args.job/'source-frames';original.mkdir(exist_ok=True)
    final=args.job/'composite';final.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(cfg['start']),'-i',str(args.source),'-vf',f"fps={cfg['fps']}",'-frames:v',str(cfg['frames']),str(original/'%03d.png')],check=True)
    frames=sorted(original.glob('*.png'))
    if len(frames)!=cfg['frames']:raise ValueError('Source segment too short')
    for i,(source,gen,mask) in enumerate(zip(frames,generated,masks)):
        alpha=Image.open(mask).convert('L').filter(ImageFilter.GaussianBlur(.7))
        result=composite(Image.open(source).convert('RGB'),Image.open(gen).convert('RGB'),alpha,cfg['crop'],cfg.get('resize_to_crop',False))
        result.save(final/f'{i:03d}.png')
    output=args.job/'replacement-preview.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(cfg['fps']),'-i',str(final/'%03d.png'),'-ss',str(cfg['start']),'-i',str(args.source),'-map','0:v:0','-map','1:a:0?','-t',str(cfg['frames']/cfg['fps']),'-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(output)],check=True)
    print(output.resolve())


if __name__=='__main__':main()
