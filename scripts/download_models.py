"""Download only pinned assets; resume partials and verify before publication."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']


def download(item, base):
    dest = base / item['destination']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not verify(dest, item):
            raise RuntimeError(f'Corrupt existing weight: {dest.name}; quarantine it before retrying')
        print('Verified existing', dest.name, flush=True)
        return
    part = dest.with_suffix(dest.suffix + '.part')
    if part.exists() and part.stat().st_size > item['size']:
        raise RuntimeError(f'Oversized partial: {part.name}')
    url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['filename']}"
    print('Downloading', dest.name, flush=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-', '--max-filesize', str(item['size']),
                    '--output', str(part), url], check=True)
    if not verify(part, item):
        raise RuntimeError(f'Checksum mismatch: {part.name}; do not load')
    part.replace(dest)
    print('Verified SHA256', dest.name, flush=True)


def remaining_bytes(manifest, base):
    total = 0
    for item in manifest:
        dest = base / item['destination']
        if dest.exists():
            continue
        part = dest.with_suffix(dest.suffix + '.part')
        total += max(0, item['size'] - (part.stat().st_size if part.exists() else 0))
    return total


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'models.lock.json').read_text())
    from resource_watch import sample
    base = ROOT / '.local/ComfyUI/models'
    additional = remaining_bytes(manifest, base)
    existing = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if existing + additional > 60_000_000_000:
        raise RuntimeError('Download would exceed 60 GB project budget')
    if sample()['available_disk'] - additional < 30_000_000_000:
        raise RuntimeError('Download would violate 30 GB available-space reserve')
    for item in manifest:
        if sample()['available_disk'] - remaining_bytes([item], base) < 30_000_000_000:
            raise RuntimeError('Available space changed; download stopped before next asset')
        download(item, base)
