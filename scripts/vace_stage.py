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
from local_paths import validate_local_output
validate_local_output(job)
config = json.loads((job / 'config.json').read_text())
animate = config.get('engine') == 'animate'
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from validation import validate_job, validate_vace_keys, flatten_video_frames, validate_motion_controls
validate_job(job, config)
if animate and stage in ('prepare', 'sample'):
    validate_motion_controls(job, config)
from download_models import verify
required = {
    'text': {'umt5-xxl-encoder-Q4_K_M.gguf'},
    'prepare': {'wan_2.1_vae.safetensors'},
    'sample': {'Wan2.1-VACE-1.3B-Q8_0.gguf', 'wan2.1_vace_1.3B_fp16.safetensors'},
    'decode': {'wan_2.1_vae.safetensors'},
    'vision': {'clip_vision_h.safetensors'},
    'roundtrip': {'wan_2.1_vae.safetensors'},
}[stage]
if animate and stage == 'sample':
    required = {config.get('model', 'Wan2.2-Animate-14B-Q2_K.gguf')}
elif stage == 'sample' and config.get('model') == 'wan2.1_vace_1.3B_fp16.safetensors':
    required = {config['model']}
for item in json.loads((ROOT / 'models.lock.json').read_text()):
    if pathlib.Path(item['filename']).name in required:
        if not verify(ROOT / '.local/ComfyUI/models' / item['destination'], item):
            raise RuntimeError('Missing or corrupt required weight: ' + pathlib.Path(item['filename']).name)
        required.remove(pathlib.Path(item['filename']).name)
if required:
    raise RuntimeError('Required weights missing from manifest')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  PYTORCH_ENABLE_MPS_FALLBACK='1',
                  HF_HOME=str(ROOT / '.cache/huggingface'),
                  TORCH_HOME=str(ROOT / '.cache/torch'))
sys.path.insert(0, str(ROOT / '.local/ComfyUI'))
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ['vace-stage', '--disable-api-nodes', '--fp32-vae', '--cpu-vae',
            '--fp16-unet', '--' + config.get('text_dtype', 'fp16') + '-text-enc', '--use-split-cross-attention']
if stage in ('text', 'prepare', 'decode', 'vision', 'roundtrip') or config.get('device') == 'cpu':
    sys.argv += ['--cpu']
import torch
torch.set_num_threads(4)
import numpy as np
from PIL import Image
import nodes
import comfy.model_management as mm
from comfy_extras.nodes_wan import WanVaceToVideo, WanAnimateToVideo
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
    result = sys.modules['local_gguf.nodes']
    original = result.gguf_sd_loader
    def checked_vace(path, *args, **kwargs):
        sd, extra = original(path, *args, **kwargs)
        if pathlib.Path(path).name == 'Wan2.1-VACE-1.3B-Q8_0.gguf':
            from vace_checkpoint import recover
            sd = recover(sd, ROOT / '.local/ComfyUI/models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors')
            logging.info('Recovered missing VACE convolution after exact unquantized-tensor comparison')
        return sd, extra
    result.gguf_sd_loader = checked_vace
    return result


def load(name):
    return torch.load(job / name, weights_only=True, map_location='cpu')


def save(name, value):
    part = job / (name + '.tmp')
    torch.save(value, part)
    part.replace(job / name)


def load_vae():
    vae = nodes.VAELoader().load_vae('wan_2.1_vae.safetensors')[0]
    if config.get('tiled_vae', False):
        # Native ComfyUI tiling; spatial units are pixels for encode, latents for decode.
        def spatial_encode(pixels):
            logging.info('Spatial VAE encode start: %s', tuple(pixels.shape))
            result = vae.encode_tiled(pixels, tile_x=256, tile_y=256, overlap=64)
            logging.info('Spatial VAE encode complete: %s', tuple(result.shape))
            return result
        def spatial_decode(latents):
            logging.info('Spatial VAE decode start: %s', tuple(latents.shape))
            result = vae.decode_tiled(latents, tile_x=32, tile_y=32, overlap=8)
            logging.info('Spatial VAE decode complete: %s', tuple(result.shape))
            return result
        vae.encode = spatial_encode
        vae.decode = spatial_decode
        logging.info('Native spatial VAE tiles: encode 256x256; decode 32x32 latent; full temporal window')
    return vae


start = time.monotonic()
print('STAGE', stage, 'START', flush=True)
with torch.inference_mode():
    if stage == 'text':
        clip = gguf_nodes().CLIPLoaderGGUF().load_clip('umt5-xxl-encoder-Q4_K_M.gguf', 'wan')[0]
        result = []
        for prompt in (config['positive'], config['negative']):
            result.append(clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))
        save('text.pt', result)
    elif stage == 'vision':
        clip = nodes.CLIPVisionLoader().load_clip('clip_vision_h.safetensors')[0]
        reference = torch.from_numpy(np.asarray(Image.open(job / 'reference.png').convert('RGB')).copy().astype(np.float32) / 255)[None]
        save('vision.pt', vars(clip.encode_image(reference, crop=False)))
    elif stage == 'prepare':
        positive, negative = load('text.pt')
        vae = load_vae()
        pixels = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in sorted((job / 'frames').glob('*.png'))]).astype(np.float32) / 255)
        masks = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('L')) for p in sorted((job / 'masks').glob('*.png'))]).astype(np.float32) / 255)
        if config.get('blank_masked_source', True):
            pixels = pixels * (1 - masks.unsqueeze(-1))
        reference = torch.from_numpy(np.asarray(Image.open(job / 'reference.png').convert('RGB')).copy().astype(np.float32) / 255)[None]
        assert len(pixels) == len(masks) == config['frames']
        if animate:
            def images(folder, count=None):
                paths = sorted((job / folder).glob('*.png'))
                if len(paths) != (config['frames'] if count is None else count):
                    raise ValueError('Incomplete motion controls: ' + folder)
                return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths]).astype(np.float32) / 255)
            continuation_count = config.get('continue_motion_frames', 0)
            continuation = images('continuation', continuation_count) if continuation_count else None
            result = WanAnimateToVideo.execute(positive, negative, vae, config['width'], config['height'], config['frames'], 1, 5, 0, reference_image=reference, face_video=images('faces'), pose_video=images('poses'), continue_motion=continuation, background_video=pixels, character_mask=masks)
            result = list(result)[:4]
        else:
            result = WanVaceToVideo.execute(positive, negative, vae, config['width'], config['height'], config['frames'], 1, config.get('strength',1.0), pixels, masks, reference)
        save('prepared.pt', list(result))
    elif stage == 'sample':
        if config.get('device') != 'cpu' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal unavailable; run in a context with GPU access')
        positive, negative, latent, trim = load('prepared.pt')
        model_name = config.get('model', 'Wan2.2-Animate-14B-Q2_K.gguf' if animate else 'Wan2.1-VACE-1.3B-Q8_0.gguf')
        class RejectIncomplete(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if message.startswith(('unet missing:', 'unet unexpected:')):
                    raise RuntimeError('Incomplete or incompatible model: ' + message)
        guard = RejectIncomplete()
        logging.getLogger().addHandler(guard)
        try:
            if model_name == 'wan2.1_vace_1.3B_fp16.safetensors':
                model = nodes.UNETLoader().load_unet(model_name, 'default')[0]
            else:
                model = gguf_nodes().UnetLoaderGGUF().load_unet(model_name, dequant_dtype='target')[0]
        finally:
            logging.getLogger().removeHandler(guard)
        expected_type = 'animate' if animate else 'vace'
        if model.model.model_config.unet_config.get('model_type') != expected_type:
            raise RuntimeError('Refusing wrong model architecture: expected ' + expected_type)
        if animate:
            from comfy.clip_vision import Output
            import node_helpers
            vision = Output()
            vars(vision).update(load('vision.pt'))
            positive = node_helpers.conditioning_set_values(positive, {'clip_vision_output': vision})
            negative = node_helpers.conditioning_set_values(negative, {'clip_vision_output': vision})
        model = ModelSamplingSD3().patch(model, config.get('shift', 5.0))[0]
        output = nodes.common_ksampler(model, config['seed'], config['steps'], config['cfg'], config.get('sampler','uni_pc'), 'simple', positive, negative, latent)[0]
        save('sampled.pt', {'samples': output['samples'][:, :, trim:].cpu()})
    elif stage in ('decode', 'roundtrip'):
        vae = load_vae()
        if stage == 'roundtrip':
            inputs = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in sorted((job / 'frames').glob('*.png'))]).astype(np.float32) / 255)
            pixels = vae.decode(vae.encode(inputs))
        else:
            pixels = vae.decode(load('sampled.pt')['samples'])
        pixels = flatten_video_frames(pixels, (config['frames'], config['height'], config['width'], 3))
        if not torch.isfinite(pixels).all():
            raise RuntimeError('Nonfinite decoded pixels')
        output = job / ('roundtrip' if stage == 'roundtrip' else 'generated'); output.mkdir(exist_ok=True)
        for i, frame in enumerate(pixels):
            Image.fromarray((frame.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(output / f'{i:03}.png')
    else:
        raise ValueError('Unknown stage')
print('STAGE', stage, 'COMPLETE', round(time.monotonic()-start,2), 'seconds', flush=True)
