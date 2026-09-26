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
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-',
                    '--output', str(part), url], check=True)
    if not verify(part, item):
        raise RuntimeError(f'Checksum mismatch: {part.name}; do not load')
    part.replace(dest)
    print('Verified SHA256', dest.name, flush=True)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'models.lock.json').read_text())
    for item in manifest:
        download(item, ROOT / '.local/ComfyUI/models')
