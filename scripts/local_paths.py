"""Reject known Dropbox output locations without an explicit local exclusion.

This is not a universal audit of backup/sync software. Prefer a nonsynced folder.
The inference child's network sandbox cannot constrain a separate sync client.
"""
from pathlib import Path
import subprocess


def validate_local_output(path):
    path = Path(path).resolve()
    ancestors = (path, *path.parents)
    roots = [i for i, p in enumerate(ancestors) if p.name.lower().startswith('dropbox')]
    if not roots:
        return
    for parent in ancestors[:max(roots) + 1]:
        for attr in ('com.dropbox.ignored', 'com.apple.fileprovider.ignore#P'):
            result = subprocess.run(['/usr/bin/xattr', '-p', attr, str(parent)],
                                    capture_output=True, check=False)
            if result.returncode == 0 and result.stdout.strip() == b'1':
                return
    raise ValueError('Output is under Dropbox without an ignore attribute. '
                     'Use a nonsynced folder or explicitly exclude the private job folder first.')
