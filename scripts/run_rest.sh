#!/usr/bin/env bash
# Remaining queue after a pause: extract (resumes from feature checkpoints), fidelity, signature ablation.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 VOICEPRINT_CONFIG=configs/session.yaml
PY=.venv/Scripts/python.exe
run() { local name=$1; shift; echo "START $(date +%T)" > scratch/$name.log; "$@" >> scratch/$name.log 2>&1; echo "EXIT $? $(date +%T)" >> scratch/$name.log; }
run extract $PY scripts/public_evals.py extract
run fidelity $PY scripts/public_evals.py fidelity
run twcs_nosig $PY scripts/public_evals.py twcs_nosig
echo "ALL DONE $(date +%T)" > scratch/queue.done
