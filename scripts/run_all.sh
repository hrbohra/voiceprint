#!/usr/bin/env bash
# Runs the full evaluation queue sequentially; each step logs to scratch/<name>.log and appends EXIT <code>.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
run() { local name=$1; shift; echo "START $(date +%T)" > scratch/$name.log; "$@" >> scratch/$name.log 2>&1; echo "EXIT $? $(date +%T)" >> scratch/$name.log; }
run planted_llm $PY -m voiceprint.cli eval planted -o results/planted_llm
run books $PY scripts/public_evals.py books
run twcs $PY scripts/public_evals.py twcs
run curve $PY scripts/public_evals.py curve
run extract $PY scripts/public_evals.py extract
run fidelity $PY scripts/public_evals.py fidelity
echo "ALL DONE $(date +%T)" > scratch/queue.done
