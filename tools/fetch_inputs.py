"""Download the pinned university archive and extract only the reserved BLIFs."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ROOT/'build/iscas-inputs'));args=p.parse_args()
    suite=json.loads((ROOT/'benchmarks/suite.json').read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    archive=out/'ISCAS.7z';expected=suite['archive']['sha256']
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
        data=urllib.request.urlopen(suite['archive']['url'],timeout=90).read()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive hash mismatch')
        archive.write_bytes(data)
    members=[r['path'] for r in suite['test']]
    for name in members:
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('Unsafe member path')
    if not all((out/r['path']).is_file() and hashlib.sha256((out/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in suite['test']):
        tool=shutil.which('bsdtar')
        if not tool:raise RuntimeError('Install libarchive bsdtar to read this .7z archive')
        subprocess.run([tool,'-xf',str(archive.resolve()),'-C',str(out.resolve()),*members],check=True)
    for row in suite['test']:
        if hashlib.sha256((out/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Extracted source hash mismatch')
    print(f'Verified {len(members)} reserved inputs. No synthesis, mapping or model evaluation performed.')


if __name__=='__main__':main()
