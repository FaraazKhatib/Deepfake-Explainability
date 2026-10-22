"""Save source snapshots or restore one into a NEW directory, without overwriting work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv','venv','node_modules','__pycache__','.pytest_cache','checkpoints','uploads','results','.git','dist'}
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['save','restore'])
parser.add_argument('name', help='Snapshot name, e.g. 02-tested')
parser.add_argument('--destination', help='New directory inside this project for a safe restore')
args = parser.parse_args()
if Path(args.name).name != args.name or not args.name.replace('-','').replace('_','').isalnum():
    parser.error('Use an alphanumeric snapshot name with optional hyphens/underscores')
archive = ROOT / 'checkpoints' / f'{args.name}.zip'
if args.action == 'save':
    archive.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for directory, names, filenames in os.walk(ROOT):
            names[:] = [name for name in names if name not in EXCLUDE]
            for filename in filenames:
                p = Path(directory) / filename
                rel = p.relative_to(ROOT)
                if p.name.startswith('.env') and p.name != '.env.example': continue
                z.write(p,rel)
                manifest[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    archive.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'{archive}: {len(manifest)} files; SHA256 {hashlib.sha256(archive.read_bytes()).hexdigest()}')
else:
    if not args.destination: parser.error('--destination is required')
    dest = (ROOT / args.destination).resolve()
    if not dest.is_relative_to(ROOT) or dest == ROOT or dest.exists():
        parser.error('Destination must be a NEW directory inside the project')
    manifest = json.loads(archive.with_suffix('.manifest.json').read_text())
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest): raise ValueError('Unsafe archive path')
            content = z.read(member)
            key = str(Path(member.filename))
            if hashlib.sha256(content).hexdigest() != manifest[key]: raise ValueError('Snapshot checksum mismatch')
        z.extractall(dest)
    print(f'Verified source restored to {dest}. Your current files were left intact.')
