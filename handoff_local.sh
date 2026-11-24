#!/usr/bin/env bash
# Optional smoke test after placing this project in its own directory.
# Reads/writes only relative to this project. Does not touch FLOW/FlowGen.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest discover -s tests -v
python3 examples/make_examples.py
python3 examples/benchmark.py
python3 flowsight.py verify reports/sprite.sqlite examples/sprite.png
python3 flowsight.py point reports/flow_hello.sqlite 2 2
