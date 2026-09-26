"""Recover the one missing VACE tensor from a verified complete checkpoint.

Do not invent weights or silently accept another architecture. The pinned Q8
community file omits this convolution. All 797 unquantized tensors must match
the complete checkpoint at FP16 before its missing tensor is used.
"""
import torch
import gguf
from safetensors import safe_open
from validation import validate_vace_keys


def recover(sd, checkpoint):
    prefix = 'model.diffusion_model.'
    key = 'vace_patch_embedding.weight'
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        expected = {k.removeprefix(prefix) for k in source.keys()}
        missing, extra = expected - set(sd), set(sd) - expected
        if missing != {key} or extra:
            raise ValueError(f'Unexpected checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
        for name, value in sd.items():
            if getattr(value, 'tensor_type', None) in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                if not torch.equal(value.to(torch.float16), source.get_tensor(prefix + name).to(torch.float16)):
                    raise ValueError(f'Checkpoint provenance mismatch: {name}')
        sd[key] = source.get_tensor(prefix + key)
    validate_vace_keys(sd)
    return sd
