#!/usr/bin/env bash
# Final queue: re-extract with exemplar dedup (cached), signature ablation (resumes from checkpoints), fidelity (session).
cd /c/dev/voiceprint
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 VOICEPRINT_CONFIG=configs/session.yaml
PY=.venv/Scripts/python.exe
run() { local name=$1; shift; echo "START $(date +%T)" > scratch/$name.log; "$@" >> scratch/$name.log 2>&1; echo "EXIT $? $(date +%T)" >> scratch/$name.log; }
run extract $PY scripts/public_evals.py extract
run twcs_nosig $PY scripts/public_evals.py twcs_nosig
run fidelity $PY scripts/public_evals.py fidelity
echo "ALL DONE $(date +%T)" > scratch/queue.done
