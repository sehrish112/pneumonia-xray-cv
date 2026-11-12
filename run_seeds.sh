#!/bin/bash
# Train + evaluate several seeds sequentially (CPU, ~15 min each at 64px / 10 epochs)
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
$PY src/evaluate.py --out outputs            # re-run seed 42 eval to save test predictions
for s in 1 2 3 4; do
  $PY src/train.py --epochs 10 --size 64 --seed $s --out outputs_seed$s
  $PY src/evaluate.py --out outputs_seed$s
done
$PY src/aggregate.py outputs outputs_seed1 outputs_seed2 outputs_seed3 outputs_seed4 > seeds_report.md
echo DONE >> seeds_report.md
