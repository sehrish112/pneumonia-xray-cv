#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
$PY src/external_validation.py --out outputs > /dev/null 2>&1
for s in 1 2 3 4; do $PY src/external_validation.py --out outputs_seed$s > /dev/null 2>&1; done
$PY src/finetune_external.py --out outputs > finetune_external.log 2>&1
echo DONE > extras_done.txt
