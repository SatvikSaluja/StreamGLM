"""New independent confirmation recipe; dry-run unless --run is supplied."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['confirmation'])
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    cmd = [sys.executable, '-u', '-m', 'streamglm.tuned_comparison', '--neurons', '64', '128', '--seeds', '59', '71', '--out', 'artifacts/confirmation_s59_s71', '--resume']
    print(shlex.join(cmd), flush=True)
    if a.run:
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

if __name__ == '__main__':
    main()
