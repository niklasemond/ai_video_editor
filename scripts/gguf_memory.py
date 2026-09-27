"""Bound temporary allocations for the UMT5 embedding expansion only.

The upstream loader already expands this table to FP16. Keep that result but
convert 256 rows at a time, instead of allocating full-table intermediates.
"""
import torch


def chunked_embedding(tensor, dequantizer, dtype=torch.float16, rows_per_chunk=256):
    rows, columns = tensor.shape
    raw = tensor.as_subclass(torch.Tensor).detach().cpu().reshape(rows, -1)
    output = torch.empty((rows, columns), dtype=dtype, device='cpu')
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        chunk = type(tensor)(raw[start:stop], tensor_type=tensor.tensor_type,
                             tensor_shape=torch.Size((stop-start, columns)))
        output[start:stop].copy_(dequantizer(chunk, dtype=dtype))
    return output


def install(loader):
    original = loader.dequantize_tensor

    def bounded(tensor, dtype=None, *args, **kwargs):
        if tuple(tensor.shape) == (256384, 4096):
            return chunked_embedding(tensor, original, dtype or torch.float16)
        return original(tensor, dtype, *args, **kwargs)

    loader.dequantize_tensor = bounded
