"""Experimental native ComfyUI VACE stages; execute each in a fresh process.

Run with the resource supervisor and a process-local network-denial profile.
No stage downloads models or invokes API nodes. Inputs/config stay in work/.
"""
import importlib.util
import json
import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
stage, job_name = sys.argv[1:3]
job = pathlib.Path(job_name).resolve()
config = json.loads((job / 'config.json').read_text())
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from validation import validate_job
validate_job(job, config)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HOME=str(ROOT / '.cache/huggingface'),
                  TORCH_HOME=str(ROOT / '.cache/torch'))
sys.path.insert(0, str(ROOT / '.local/ComfyUI'))
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ['vace-stage', '--disable-api-nodes', '--fp32-vae', '--cpu-vae',
            '--fp16-unet', '--fp16-text-enc', '--use-split-cross-attention']
if stage in ('text', 'prepare', 'decode'):
    sys.argv += ['--cpu']
import torch
torch.set_num_threads(4)
import numpy as np
from PIL import Image
import nodes
import comfy.model_management as mm
from comfy_extras.nodes_wan import WanVaceToVideo
from comfy_extras.nodes_model_advanced import ModelSamplingSD3


def gguf_nodes():
    path = ROOT / '.local/ComfyUI-GGUF'
    spec = importlib.util.spec_from_file_location('local_gguf', path / '__init__.py',
                                                 submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from gguf_memory import install
    install(sys.modules['local_gguf.loader'])
    return sys.modules['local_gguf.nodes']


def load(name):
    return torch.load(job / name, weights_only=True, map_location='cpu')


def save(name, value):
    part = job / (name + '.tmp')
    torch.save(value, part)
    part.replace(job / name)


start = time.monotonic()
print('STAGE', stage, 'START', flush=True)
with torch.inference_mode():
    if stage == 'text':
        clip = gguf_nodes().CLIPLoaderGGUF().load_clip('umt5-xxl-encoder-Q4_K_M.gguf', 'wan')[0]
        result = []
        for prompt in (config['positive'], config['negative']):
            result.append(clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))
        save('text.pt', result)
    elif stage == 'prepare':
        positive, negative = load('text.pt')
        vae = nodes.VAELoader().load_vae('wan_2.1_vae.safetensors')[0]
        pixels = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in sorted((job / 'frames').glob('*.png'))]).astype(np.float32) / 255)
        masks = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('L')) for p in sorted((job / 'masks').glob('*.png'))]).astype(np.float32) / 255)
        reference = torch.from_numpy(np.asarray(Image.open(job / 'reference.png').convert('RGB')).copy().astype(np.float32) / 255)[None]
        assert len(pixels) == len(masks) == config['frames']
        result = WanVaceToVideo.execute(positive, negative, vae, config['width'], config['height'], config['frames'], 1, config.get('strength',1.0), pixels, masks, reference)
        save('prepared.pt', list(result))
    elif stage == 'sample':
        if not torch.backends.mps.is_available():
            raise RuntimeError('Metal unavailable; run in a context with GPU access')
        positive, negative, latent, trim = load('prepared.pt')
        model = gguf_nodes().UnetLoaderGGUF().load_unet('Wan2.1-VACE-1.3B-Q8_0.gguf', dequant_dtype='target')[0]
        model = ModelSamplingSD3().patch(model, config.get('shift', 5.0))[0]
        output = nodes.common_ksampler(model, config['seed'], config['steps'], config['cfg'], 'uni_pc', 'simple', positive, negative, latent)[0]
        save('sampled.pt', {'samples': output['samples'][:, :, trim:].cpu()})
    elif stage == 'decode':
        vae = nodes.VAELoader().load_vae('wan_2.1_vae.safetensors')[0]
        pixels = vae.decode(load('sampled.pt')['samples'])
        if not torch.isfinite(pixels).all():
            raise RuntimeError('Nonfinite decoded pixels')
        output = job / 'generated'; output.mkdir(exist_ok=True)
        for i, frame in enumerate(pixels):
            Image.fromarray((frame.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(output / f'{i:03}.png')
    else:
        raise ValueError('Unknown stage')
print('STAGE', stage, 'COMPLETE', round(time.monotonic()-start,2), 'seconds', flush=True)
