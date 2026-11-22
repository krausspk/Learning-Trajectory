#!/usr/bin/env python3
"""Verify all finalized research artifact fingerprints using the standard library."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'outputs'))
    args=parser.parse_args();root=Path(args.output)
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    errors=[]
    for entry in manifest['files']:
        path=root/entry['path']
        if not path.is_file():
            errors.append(f"Missing: {entry['path']}");continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        if path.stat().st_size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
            errors.append(f"Changed: {entry['path']}")
    result={'checked_files':len(manifest['files']),'errors':errors,'passed':not errors}
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
