#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.pipelines.object_level import run_from_file


def main():
    parser = argparse.ArgumentParser(description='Run V2X calibration pipeline')
    parser.add_argument('--config', type=str, default='configs/pipeline.yaml')
    parser.add_argument('--print', action='store_true', help='Print summary to stdout')
    args = parser.parse_args()
    summary = run_from_file(args.config)
    if args.print:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
