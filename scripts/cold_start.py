"""Download NASA data afresh and execute one complete CPU benchmark.

Network access is explicit here, never implicit in the package's train command.
The archive is retained locally for provenance; do not upload raw NASA files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

URL = 'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dataset', choices=['FD001', 'FD003'], required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(dataset=args.dataset, source_url=URL, seed=42, streams=[9001], bootstrap_reps=2000,
                    python=sys.version, commands=[], status='RUNNING')
    try:
        archive = out / 'NASA_CMAPSS.zip'
        request = urllib.request.Request(URL, headers={'User-Agent': 'RiskSelRUL-cold-start/0.1.1'})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open('xb') as dest:
            manifest['resolved_url'] = response.url
            while chunk := response.read(1024*1024):
                dest.write(chunk)
        manifest['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest['archive_bytes'] = archive.stat().st_size
        commands = [
            ['extract-data', '--zip', str(archive), '--out', str(out/'data')],
            ['inspect-data', '--data-root', str(out/'data'), '--dataset', args.dataset, '--strict-hashes'],
            ['reproduce', '--data-root', str(out/'data'), '--dataset', args.dataset,
             '--seed', '42', '--streams', '9001', '--bootstrap-reps', '2000', '--out', str(out/'run')],
            ['audit', '--run', str(out/'run')],
        ]
        for index, arguments in enumerate(commands):
            command = [sys.executable, '-m', 'riskselrul', *arguments]
            tick = time.time()
            with (out/f'{index:02d}.stdout.log').open('x', encoding='utf-8') as stdout, \
                    (out/f'{index:02d}.stderr.log').open('x', encoding='utf-8') as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr)
            manifest['commands'].append(dict(argv=command, exitcode=process.returncode, seconds=time.time()-tick))
            if process.returncode:
                raise RuntimeError(f'Command {index} exited {process.returncode}')
        manifest['status'] = 'PASS'
    except Exception as exc:
        manifest['status'] = 'FAIL'
        manifest['error'] = str(exc)
        raise
    finally:
        manifest['seconds'] = time.time()-started
        (out/'cold_start_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
